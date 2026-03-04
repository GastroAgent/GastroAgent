import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt
from skimage import io, color, measure, img_as_float, img_as_ubyte
from skimage.segmentation import slic
import cv2
import ot
from matplotlib.patches import Rectangle
import os
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from sklearn.metrics import pairwise_distances

# 辅助函数：简化目录创建
def mkdir_if_not_exist(path):
    os.makedirs(path, exist_ok=True)

# 设置中文字体
def set_chinese_font():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.ioff()  # 非交互模式，避免图像显示问题

# 为每个线程创建独立的日志记录器
def get_thread_logger(log_path, thread_id):
    logger = logging.getLogger(f"thread_{thread_id}")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# 日志配置（主进程）
def setup_main_logging(log_path):
    mkdir_if_not_exist(os.path.dirname(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger("main")

# 加载深度图
def load_depth_map(img_path, depth_root):
    try:
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(depth_root, f"{img_basename}_depth.png")
        
        if not os.path.exists(depth_path):
            # 深度图缺失时，用图像灰度值模拟深度
            img = io.imread(img_path)
            gray = color.rgb2gray(img)
            return cv2.GaussianBlur(gray, (5,5), 0)
        
        depth = io.imread(depth_path, as_gray=True)
        depth = cv2.medianBlur(depth, 3)  # 深度图去噪
        return img_as_float(depth)
    except Exception as e:
        return None

# 计算边缘区域权重
def calculate_edge_weights(image_shape, edge_ratio=0.1):
    h, w = image_shape[:2]
    edge_width = max(2, int(min(h, w) * edge_ratio))  # 基于图像比例计算边缘宽度
    
    weights = np.ones((h, w), dtype=np.float32)
    # 边缘区域权重降低（仅极窄边缘）
    weights[:edge_width, :] = 0.3  # 顶部边缘
    weights[-edge_width:, :] = 0.3  # 底部边缘
    weights[:, :edge_width] = 0.3  # 左侧边缘
    weights[:, -edge_width:] = 0.3  # 右侧边缘
    
    return weights

# 超像素处理（提高分块数目，更精细分割）
def process_superpixels(image_rgb, depth_map, img_shape):
    try:
        # 提高超像素分块数目：从每15000像素一个改为每8000像素一个
        # 同时增加最小分块数，确保细节捕捉
        h, w = img_shape[:2]
        area = h * w
        # 调整超像素数量计算方式，提高分块密度
        # num_segments = max(30, min(200, int(area / 8000)))  # 关键调整：每8000像素一个超像素
        num_segments = max(30, min(200, int(area / 6000)))
        # 紧凑度随分块数增加而降低，保持分割精细度
        # compactness = max(20, min(25, int((h + w) / 250)))   # 更精细的紧凑度调整
        compactness = max(20, min(25, int((h + w) / 200)))

        segments = slic(
            img_as_float(image_rgb),
            n_segments=num_segments,
            compactness=compactness,
            start_label=1,
            sigma=1.5,  # 轻微增加平滑，平衡精细度和噪声
            channel_axis=-1
        )
        
        # 计算边缘权重
        edge_weights = calculate_edge_weights(img_shape)
        
        # 深度权重计算
        depth_weights = {}
        edge_region_weights = {}
        for sp_id in np.unique(segments):
            mask = segments == sp_id
            avg_depth = np.mean(depth_map[mask]) if np.any(mask) else 0.5
            
            # 深度权重（更平滑的过渡）
            if avg_depth > 0.7:  # 深深度
                depth_weights[sp_id] = 0.5
            elif avg_depth < 0.3:  # 浅深度
                depth_weights[sp_id] = 1.0 + (0.3 - avg_depth) * 0.8
            else:  # 中间深度
                depth_weights[sp_id] = 1.0
            
            # 边缘区域权重
            edge_weight = np.mean(edge_weights[mask])
            edge_region_weights[sp_id] = edge_weight
        
        # 特征提取
        gray = color.rgb2gray(image_rgb)
        grad = np.sqrt(
            cv2.Sobel(img_as_ubyte(gray), cv2.CV_64F, 1, 0, ksize=3)**2 +
            cv2.Sobel(img_as_ubyte(gray), cv2.CV_64F, 0, 1, ksize=3)** 2
        )
        
        features = []
        for sp_id in np.unique(segments):
            mask = segments == sp_id
            sp_hsv = color.rgb2hsv(image_rgb)[mask]
            props = measure.regionprops(measure.label(mask))[0]
            r1, r2 = np.min(np.where(mask)[0]), np.max(np.where(mask)[0])
            c1, c2 = np.min(np.where(mask)[1]), np.max(np.where(mask)[1])
            
            # 综合权重 = 深度权重 * 边缘区域权重
            combined_weight = depth_weights[sp_id] * edge_region_weights[sp_id]
            
            features.append({
                'id': sp_id, 'mask': mask, 'area': np.sum(mask),
                'solidity': props.solidity, 'centroid': props.centroid,
                'mean_hsv': np.mean(sp_hsv, axis=0), 'std_hsv': np.std(sp_hsv, axis=0),
                'mean_gradient': np.mean(grad[mask]) if np.any(mask) else 0,
                'bbox': (r1, r2, c1, c2), 'combined_weight': combined_weight,
                'pixels': image_rgb[mask].astype(np.uint8)
            })
        return features, segments, num_segments, compactness
    except Exception as e:
        print(f"超像素处理错误: {str(e)}")
        return None, None, 0, 0

# 异常分数计算（优化自适应阈值）
def calculate_anomaly_scores(features, reg=0.08):
    try:
        vectors = np.array([
            np.concatenate([f['mean_hsv'], f['std_hsv'], [f['mean_gradient']], [f['solidity']]])
            for f in features
        ])
        
        # 优化距离计算
        cost_matrix = pairwise_distances(vectors, metric="euclidean")
        n = len(vectors)
        a = np.ones(1)/1
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            dist_matrix[i] = ot.sinkhorn2(a, np.ones(n)/n, cost_matrix[i:i+1], reg)
        
        base_scores = np.mean(dist_matrix, axis=1)
        final_scores = base_scores * np.array([f['combined_weight'] for f in features])
        
        # 优化自适应阈值：使用更灵活的计算方式（考虑数据分布）
        q75, q25 = np.percentile(final_scores, [75 ,25])
        iqr = q75 - q25  # 四分位距，更稳健的离散度指标
        adaptive_threshold = q75 + 1.2 * iqr  # 基于四分位距的阈值，减少极端值影响
        
        return base_scores, final_scores, adaptive_threshold
    except Exception as e:
        print(f"异常分数计算错误: {str(e)}")
        return None, None, 0

# 异常区域验证（放宽过度严格的筛选条件）
def is_valid(sp, sim_thresh=25, max_sim=0.8, black_thresh=35, max_black=0.08):
    """放宽验证条件，减少有效区域误判"""
    try:
        # 放宽最小区域限制（从10像素改为5像素）
        if len(sp['pixels']) < 5:
            return False, "区域过小"
        
        sample_idx = np.random.choice(len(sp['pixels']), min(10, len(sp['pixels'])), replace=False)
        sim_mean = np.mean([
            np.mean(np.sqrt(np.sum((sp['pixels'] - sp['pixels'][i])**2, axis=1)) < sim_thresh)
            for i in sample_idx
        ])
        # 放宽相似性限制（从0.75改为0.8）
        if sim_mean >= max_sim:
            return False, f"相似占比{sim_mean:.0%}>{max_sim:.0%}"
        
        # 放宽黑色像素限制（从0.05改为0.08）
        black_pct = np.mean(np.all(sp['pixels'] < black_thresh, axis=1))
        if black_pct >= max_black:
            return False, f"黑色占比{black_pct:.0%}>{max_black:.0%}"
        
        return True, "有效"
    except Exception as e:
        return False, f"验证错误: {str(e)}"

# 寻找连通的异常区域（优化聚类逻辑）
def find_connected_regions(valid_sps, distance_threshold=None):
    """优化连通区域判断，确保更多有效区域被纳入"""
    try:
        if not valid_sps:
            return []
        
        # 按分数排序
        valid_sps_sorted = sorted(valid_sps, key=lambda x: x['final_score'], reverse=True)
        groups = []
        
        # 动态距离阈值：基于图像中所有超像素的平均大小自动计算
        if distance_threshold is None and valid_sps_sorted:
            avg_size = np.mean([np.sqrt(s['area']) for s in valid_sps_sorted])
            distance_threshold = max(30, int(avg_size * 2.5))  # 自适应距离阈值
        
        # 聚类相似区域：允许更多区域被纳入同一组
        for sp in valid_sps_sorted:
            added = False
            # 优先加入已有成员最多的组（提高聚类效率）
            for group in sorted(groups, key=lambda x: len(x), reverse=True):
                # 检查与组内任何区域是否接近
                for member in group:
                    c1, c2 = sp['centroid'], member['centroid']
                    distance = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])** 2)
                    if distance < distance_threshold:
                        group.append(sp)
                        added = True
                        break
                if added:
                    break
            if not added:
                groups.append([sp])
        
        # 选择最大的连通组，若最大组过小则合并相近小组
        if groups:
            largest_group = max(groups, key=lambda x: len(x))
            
            # 合并接近最大组的小组（避免遗漏邻近区域）
            if len(largest_group) < 3:  # 若最大组较小
                for group in groups:
                    if group is not largest_group and len(group) > 0:
                        # 检查小组与最大组的距离
                        min_dist = min(np.sqrt((sp1['centroid'][0]-sp2['centroid'][0])**2 + 
                                              (sp1['centroid'][1]-sp2['centroid'][1])** 2)
                                      for sp1 in largest_group for sp2 in group)
                        if min_dist < distance_threshold * 1.5:  # 放宽合并阈值
                            largest_group.extend(group)
            
            return largest_group if largest_group else [valid_sps_sorted[0]]
        return [valid_sps_sorted[0]]  # 确保至少返回一个区域
    except Exception as e:
        print(f"寻找连通区域错误: {str(e)}")
        return []

# 裁剪区域确定（优化边界计算）
def get_crop_region(valid_sps):
    try:
        if not valid_sps:
            return None, []
        
        # 计算最小外接矩形，确保所有有效区域被包含
        r_coords = []
        c_coords = []
        for s in valid_sps:
            r1, r2 = s['bbox'][0], s['bbox'][1]
            c1, c2 = s['bbox'][2], s['bbox'][3]
            r_coords.extend([r1, r2])
            c_coords.extend([c1, c2])
        
        r1, r2 = min(r_coords), max(r_coords)
        c1, c2 = min(c_coords), max(c_coords)
        return (r1, r2, c1, c2), valid_sps
    except Exception as e:
        print(f"裁剪区域确定错误: {str(e)}")
        return None, []

# 保存裁剪图像（放宽大小限制）
def save_crop_image(image, valid_sps, save_path, logger, crop_type="masked"):
    try:
        crop_bbox, selected = get_crop_region(valid_sps)
        if not crop_bbox:
            return None
        
        # 动态边界扩展：根据区域大小调整扩展幅度
        r1, r2, c1, c2 = crop_bbox
        region_size = (r2 - r1) * (c2 - c1)
        expand = max(5, min(15, int(np.sqrt(region_size) * 0.1)))  # 按区域大小的10%扩展
        r1, r2 = max(0, r1-expand), min(image.shape[0], r2+expand)
        c1, c2 = max(0, c1-expand), min(image.shape[1], c2+expand)
        
        # 放宽过小区域限制（从10像素改为5像素）
        if (r2-r1) < 5 or (c2-c1) < 5:
            logger.warning(f"跳过过小裁剪区: {os.path.basename(save_path)}")
            return None
        
        # 提取裁剪区域
        crop_roi = image[r1:r2, c1:c2].copy()
        
        if crop_type == "masked":
            mask_roi = np.zeros((r2-r1, c2-c1), dtype=bool)
            for sp in selected:
                sp_mask_roi = sp['mask'][r1:r2, c1:c2]
                mask_roi |= sp_mask_roi
            
            crop_roi = cv2.cvtColor(crop_roi, cv2.COLOR_RGB2BGR)
            alpha = np.ones((r2-r1, c2-c1, 1), dtype=np.uint8) * 255
            alpha[~mask_roi] = 77  # 半透明
            crop_roi_with_alpha = cv2.merge((crop_roi, alpha))
            crop_final = cv2.cvtColor(crop_roi_with_alpha, cv2.COLOR_BGRA2RGB)
        
        elif crop_type == "local":
            crop_final = crop_roi
        
        else:
            logger.error(f"不支持的裁剪类型: {crop_type}")
            return None
        
        # 保存图像
        io.imsave(save_path, crop_final)
        logger.info(f"保存{crop_type}类型裁剪图(含{len(selected)}超像素): {os.path.basename(save_path)}")
        return save_path
    except Exception as e:
        logger.error(f"保存裁剪图失败: {str(e)}")
        return None

# 结果可视化（优化有效区域筛选）
def visualize_results(image, features, output_path, crop_path, logger, crop_type="masked", 
                      score_mode="adaptive", top_n=8, filter_params={}):
    """优化有效区域筛选逻辑，避免遗漏"""
    try:
        set_chinese_font()
        final_scores = np.array([f['final_score'] for f in features])
        
        # 确定候选异常区：扩大候选范围
        if score_mode == "adaptive":
            threshold = np.array([f['adaptive_threshold'] for f in features])[0]
            candidate_indices = np.where(final_scores > threshold)[0]
            # 确保至少有候选区域，若不足则补充高分区域
            if len(candidate_indices) < 3:
                top_indices = np.argsort(final_scores)[-5:][::-1]
                candidate_indices = np.unique(np.concatenate([candidate_indices, top_indices]))
        else:  # topk模式：增加候选数量（从5到8）
            candidate_indices = np.argsort(final_scores)[-top_n:][::-1]
        
        # 筛选有效异常区：取消数量硬限制，收集所有有效区域
        valid_sps = []
        for idx in candidate_indices:
            valid, reason = is_valid(features[idx], **filter_params)
            if valid:
                valid_sps.append(features[idx])
            else:
                logger.debug(f"区域 {features[idx]['id']} 无效: {reason}")
        
        # 确保至少有一个异常区域被检测出
        if not valid_sps:
            logger.warning("未找到有效异常区域，强制选择分数最高的3个区域")
            # 强制选择分数最高的3个区域，即使它们未通过验证
            top_indices = np.argsort(final_scores)[-3:][::-1]
            valid_sps = [features[idx] for idx in top_indices]
        
        # 找到连通的异常区域（优化版）
        connected_sps = find_connected_regions(valid_sps)
        
        # 保存裁剪图
        save_crop_image(image, connected_sps, crop_path, logger, crop_type)
        
        # 绘制结果图
        mkdir_if_not_exist(os.path.dirname(output_path))
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(image)  # 使用RGB原图作为背景
        crop_bbox, selected = get_crop_region(connected_sps)
        
        # 标记候选异常区
        for rank, idx in enumerate(candidate_indices, 1):
            sp = features[idx]
            r1, r2, c1, c2 = sp['bbox']
            is_sel = any(s['id'] == sp['id'] for s in selected)
            color = 'red' if is_sel else 'orange'  # 鲜艳的锚框颜色
            
            # 掩膜叠加
            mask_vis = np.zeros_like(image)
            mask_vis[sp['mask']] = [255,0,0] if is_sel else [255,165,0]
            ax.imshow(mask_vis, alpha=0.5 if is_sel else 0.3)
            
            # 边界框与文字标注
            ax.add_patch(Rectangle((c1, r1), c2-c1, r2-r1, linewidth=3, edgecolor=color, facecolor='none'))
            status = "选中区域" if is_sel else f"无效: {is_valid(sp)[1]}"
            ax.text(c1, r1-20, f'异常 #{rank}\n分数: {sp["final_score"]:.4f}\n{status}',
                    color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.9))
        
        # 标记裁剪区
        if crop_bbox:
            r1, r2, c1, c2 = crop_bbox
            ax.add_patch(Rectangle((c1, r1), c2-c1, r2-r1, linewidth=4, edgecolor='blue', 
                                  linestyle='--', facecolor='none'))  # 鲜艳的蓝色
            ax.text(c1, r1-40, f'裁剪区域(含{len(selected)}超像素, 类型:{crop_type})', 
                    color='blue', fontsize=11, weight='bold', bbox=dict(facecolor='white', alpha=0.9))
        
        ax.set_title('胃镜图像异常检测结果', fontsize=14)
        ax.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # 确保关闭图像，释放内存
        return len(valid_sps) > 0
    except Exception as e:
        logger.error(f"可视化结果错误: {str(e)}")
        return False

# 单图像处理
def process_image(img_path, depth_root, output_path, crop_path, params, filter_params, 
                  log_path, crop_type="masked", score_mode="adaptive", thread_id=0):
    # 为每个线程创建独立的logger
    logger = get_thread_logger(log_path, thread_id)
    
    try:
        # 加载原图
        img = io.imread(img_path)
        if img.ndim != 3:
            return {'status': '跳过', '原因': '非RGB图像', '图像路径': img_path}
        
        # 加载深度图
        depth_map = load_depth_map(img_path, depth_root)
        if depth_map is None:
            logger.warning(f"深度图处理失败: {os.path.basename(img_path)}")
            return {'status': '跳过', '原因': '深度图处理失败', '图像路径': img_path}
        
        # 超像素与异常分数计算
        features, _, num_segments, compactness = process_superpixels(img, depth_map, img.shape)
        if features is None:
            return {'status': '失败', '原因': '超像素处理失败', '图像路径': img_path}
        
        base_scores, final_scores, adaptive_threshold = calculate_anomaly_scores(features, params['reg'])
        if base_scores is None:
            return {'status': '失败', '原因': '异常分数计算失败', '图像路径': img_path}
        
        for i, f in enumerate(features):
            f['base_score'], f['final_score'] = base_scores[i], final_scores[i]
            f['adaptive_threshold'] = adaptive_threshold
        
        # 可视化与结果返回
        has_valid = visualize_results(
            img, features, output_path, crop_path, logger, crop_type, 
            score_mode, params['top_n'], filter_params
        )
        
        return {
            'status': '成功', 
            '有效异常': has_valid, 
            '裁剪类型': crop_type,
            '超像素数量': num_segments,
            '紧凑度参数': compactness,
            '图像路径': img_path,
            '平均异常分数': np.mean(final_scores),
            '最高异常分数': np.max(final_scores)
        }
    
    except Exception as e:
        logger.error(f"处理失败 {os.path.basename(img_path)}: {str(e)}")
        return {'status': '失败', '原因': str(e), '图像路径': img_path}

# 批量处理主函数
def batch_process(input_root, depth_root, output_root, crop_root, params, filter_params, 
                 crop_type="masked", score_mode="adaptive", mode="full", test_count=10):
    # 初始化配置
    mkdir_if_not_exist(output_root)
    mkdir_if_not_exist(crop_root)
    log_path = os.path.join(output_root, 'logs', 'detection.log')
    main_logger = setup_main_logging(log_path)
    set_chinese_font()
    
    # 检查关键目录
    for path in [input_root, depth_root]:
        if not os.path.exists(path):
            main_logger.error(f"目录不存在: {path}")
            return
    
    # 检查参数合法性
    if crop_type not in ["masked", "local"]:
        main_logger.error(f"裁剪类型错误: {crop_type}，仅支持'masked'或'local'")
        return
    
    if score_mode not in ["adaptive", "topk"]:
        main_logger.error(f"分数模式错误: {score_mode}，仅支持'adaptive'或'topk'")
        return
    
    # 获取所有图像路径
    img_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_root)
        for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    
    if not img_paths:
        main_logger.error("未找到任何图像文件")
        return
    
    # 测试模式：仅处理固定数量的图片
    if mode == "test":
        img_paths = img_paths[:test_count]
        main_logger.info(f"测试模式：处理 {len(img_paths)} 张图像")
    else:
        main_logger.info(f"全量模式：处理 {len(img_paths)} 张图像")
    
    main_logger.info(f"深度图目录: {depth_root}，裁剪类型: {crop_type}，分数模式: {score_mode}")
    
    # 批量处理
    results = []
    max_workers = min(os.cpu_count(), 4)  # 限制线程数量，避免资源竞争
    main_logger.info(f"使用 {max_workers} 个线程进行处理")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for idx, img_path in enumerate(img_paths):
            rel_path = os.path.relpath(img_path, input_root)
            fname, ext = os.path.splitext(rel_path)
            out_img = os.path.join(output_root, f"{fname}_anomaly{ext}")
            out_crop = os.path.join(crop_root, f"{fname}_{crop_type}_crop{ext}")
            
            mkdir_if_not_exist(os.path.dirname(out_img))
            mkdir_if_not_exist(os.path.dirname(out_crop))
            
            # 提交任务
            future = executor.submit(
                process_image, img_path, depth_root, out_img, out_crop,
                params, filter_params, log_path, crop_type, score_mode, idx
            )
            futures[future] = img_path
        
        # 获取结果（带进度条）
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                main_logger.error(f"获取结果时出错: {str(e)}")
    
    # 结果统计增强
    success_cnt = sum(1 for r in results if r['status'] == '成功')
    valid_cnt = sum(1 for r in results if r.get('有效异常', False))
    skip_cnt = sum(1 for r in results if r['status'] == '跳过')
    fail_cnt = sum(1 for r in results if r['status'] == '失败')
    
    # 生成详细统计报表
    report_df = pd.DataFrame(results)
    report_path = os.path.join(output_root, '处理结果报表.xlsx')
    try:
        report_df.to_excel(report_path, index=False, engine='openpyxl')
    except Exception as e:
        main_logger.error(f"保存报表失败: {str(e)}")
    
    # 汇总统计信息
    stats = {
        '总处理图像': len(results),
        '成功处理': success_cnt,
        '有效异常区域': valid_cnt,
        '跳过处理': skip_cnt,
        '处理失败': fail_cnt,
        '平均超像素数量': report_df[report_df['status'] == '成功']['超像素数量'].mean() if success_cnt > 0 else 0,
        '平均异常分数': report_df[report_df['status'] == '成功']['平均异常分数'].mean() if success_cnt > 0 else 0
    }
    
    # 打印统计信息
    main_logger.info("\n" + "="*50)
    main_logger.info("处理统计结果")
    main_logger.info("="*50)
    for key, value in stats.items():
        main_logger.info(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    main_logger.info(f"结果报表已保存至: {report_path}")
    main_logger.info(f"配置信息: 裁剪类型={crop_type}, 分数模式={score_mode}, 模式={mode}")

if __name__ == "__main__":
    # 参数配置
    params = {
        'reg': 0.08,           # Sinkhorn正则化参数
        'top_n': 6             # 增加topk模式下的候选数量
    }
    
    filter_params = {
        'sim_thresh': 15,      # 放宽像素相似阈值
        'max_sim': 0.75,        # 放宽最大相似占比
        'black_thresh': 35,    # 放宽黑色像素阈值
        'max_black': 0.08      # 放宽最大黑色占比
    }
    
    labels = os.listdir("./Datasets/mini-imagenet-folder")
    # labels = ["胃窦溃疡（S1期）v1"]
    for label in labels: 
        print(label)
        # 路径配置（请替换为实际路径）
        input_root = f"./Datasets/mini-imagenet-folder/{label}"
        depth_root = f"./Datasets/depth/mini-imagenet-folder/{label}"
        output_root = f"./Datasets/unused/{label}"
        crop_root = f"./Datasets/mini-imagenet-folder-region/{label}"
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(crop_root, exist_ok=True)

        # 处理模式配置
        mode = "full"  # "test" 测试模式, "full" 全量模式
        test_count = 30  # 测试模式下处理的图片数量
        crop_type = "masked"  # "masked" 掩膜图, "local" 局部区域图
        score_mode = "adaptive"  # "adaptive" 自适应阈值, "topk" 前k个
                    
        # 启动批量处理
        batch_process(
            input_root, depth_root, output_root, crop_root,
            params, filter_params, crop_type, score_mode, mode, test_count
        )
