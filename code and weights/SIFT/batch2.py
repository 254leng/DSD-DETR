import numpy as np
import cv2
import os
import datetime
import pickle
import json
from typing import List, Optional, Tuple

class EnhancedTransformReuseStitcher:
    def __init__(self, feature_detector='SIFT'):
        """
        初始化拼接器
        
        Args:
            feature_detector: 'SIFT', 'ORB', 或 'AKAZE'
        """
        self.feature_detector_type = feature_detector
        self.feature_detector = self._create_feature_detector(feature_detector)
        self.saved_transform = None
        self.transform_params = None
        self.transform_chain = []  # 存储变换链
        self.sequential_transforms = []  # 新增：存储顺序拼接的变换链
    
    def _create_feature_detector(self, detector_type):
        """创建特征检测器"""
        if detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=5000)
        elif detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"不支持的检测器类型: {detector_type}")
    
    def stitch_multiple_images_sequential(self, image_folder: str, output_path: str = None,
                                        save_intermediate: bool = False,
                                        intermediate_folder: str = None,
                                        save_transforms: bool = True,
                                        transforms_save_path: str = None) -> np.ndarray:
        """
        顺序拼接文件夹中的多张图像成一张全景图
        
        Args:
            image_folder: 图像文件夹路径
            output_path: 最终结果保存路径
            save_intermediate: 是否保存中间步骤结果
            intermediate_folder: 中间结果保存文件夹
            save_transforms: 是否保存变换参数链  # 新增
            transforms_save_path: 变换参数保存路径  # 新增
            
        Returns:
            拼接后的全景图像
        """
        if not os.path.exists(image_folder):
            raise ValueError(f"文件夹不存在: {image_folder}")
        
        # 获取所有图像文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        # image_files = sorted([f for f in os.listdir(image_folder) 
        #                     if f.lower().endswith(image_extensions)])
        filelist = os.listdir(image_folder)
        image_files = sorted(
            [f for f in filelist if f.lower().endswith(image_extensions)],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        
        if len(image_files) < 2:
            raise ValueError(f"文件夹中图像文件少于2张，无法进行拼接")
        
        print(f"找到 {len(image_files)} 张图像，开始顺序拼接...")
        
        # 创建中间结果保存文件夹
        if save_intermediate and intermediate_folder:
            os.makedirs(intermediate_folder, exist_ok=True)
        
        # 读取第一张图像作为基础
        base_image = cv2.imread(os.path.join(image_folder, image_files[0]))
        if base_image is None:
            raise ValueError(f"无法读取第一张图像: {image_files[0]}")
        
        print(f"基础图像: {image_files[0]} - 尺寸: {base_image.shape}")
        current_result = base_image.copy()

        # 清空变换链
        self.sequential_transforms = []
        base_size = (int(base_image.shape[1]), int(base_image.shape[0]))  # 转换为Python int
        

        # 逐张拼接后续图像
        for i in range(1, len(image_files)):
            print(f"\n正在拼接第 {i+1} 张图像: {image_files[i]}")
            
            next_image = cv2.imread(os.path.join(image_folder, image_files[i]))
            if next_image is None:
                print(f"警告: 无法读取图像 {image_files[i]}，跳过")
                continue
            
            # 执行拼接
            result = self.stitch_with_feature_matching(
                [next_image, current_result],  # [imageB, imageA]
                showMatches=False,
                save_transform=True
            )
            
            if result is not None:
            # 保存当前步骤的变换信息 - 转换numpy类型为Python原生类型
                if save_transforms and self.saved_transform is not None:
                    step_transform = {
                        'step': int(i),  # 确保是Python int
                        'source_image': image_files[i],
                        'homography_matrix': self.saved_transform.tolist(),
                        'transform_params': {
                            'translation_dist': [int(x) for x in self.transform_params['translation_dist']],
                            'result_width': int(self.transform_params['result_width']),
                            'result_height': int(self.transform_params['result_height']),
                            'H_translation': self.transform_params['H_translation'].tolist(),
                            'original_imageA_size': [int(x) for x in self.transform_params['original_imageA_size']],
                            'original_imageB_size': [int(x) for x in self.transform_params['original_imageB_size']]
                        },
                        'current_result_size': [int(result.shape[1]), int(result.shape[0])]  # 转换为Python int
                    }
                    self.sequential_transforms.append(step_transform)

                current_result = result
                print(f"拼接成功，当前结果尺寸: {current_result.shape}")
                
                # 保存中间结果
                if save_intermediate and intermediate_folder:
                    intermediate_path = os.path.join(intermediate_folder, 
                                                   f"step_{i:02d}_added_{image_files[i]}")
                    cv2.imwrite(intermediate_path, current_result)
                    print(f"中间结果已保存: {intermediate_path}")
            else:
                print(f"警告: 拼接失败 {image_files[i]}，跳过此图像")
                continue

        
        # 保存变换参数链
        if save_transforms and self.sequential_transforms:
            if transforms_save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                transforms_save_path = f"sequential_transforms_{timestamp}.json"
            
            transform_data = {
                'metadata': {
                    'creation_time': datetime.datetime.now().isoformat(),
                    'base_image': image_files[0],
                    'base_image_size': base_size,
                    'total_images': int(len(image_files)),  # 确保是Python int
                    'successful_steps': int(len(self.sequential_transforms)),  # 确保是Python int
                    'feature_detector': self.feature_detector_type
                },
                'transforms': self.sequential_transforms
            }
            
            # 确保保存路径的目录存在
            if transforms_save_path and os.path.dirname(transforms_save_path):
                os.makedirs(os.path.dirname(transforms_save_path), exist_ok=True)
            
            try:
                with open(transforms_save_path, 'w', encoding='utf-8') as f:
                    json.dump(transform_data, f, indent=2, ensure_ascii=False)
                
                print(f"变换参数链已保存到: {transforms_save_path}")
            except Exception as e:
                print(f"保存变换参数时出错: {e}")
                print("可能的原因: 数据类型转换问题")
    
        
        # 保存最终结果
        if output_path:
            if os.path.dirname(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            success = cv2.imwrite(output_path, current_result)
            if success:
                print(f"\n最终全景图已保存到: {output_path}")
            else:
                print(f"保存失败: {output_path}")
        
        return current_result
    
    def stitch_multiple_images_pairwise(self, image_folder: str, output_folder: str,
                                      use_saved_transform: bool = False,
                                      transform_file: str = None) -> List[str]:
        """
        成对拼接文件夹中的图像
        
        Args:
            image_folder: 输入图像文件夹
            output_folder: 输出文件夹
            use_saved_transform: 是否使用保存的变换
            transform_file: 变换参数文件路径
            
        Returns:
            成功拼接的文件列表
        """
        if not os.path.exists(image_folder):
            raise ValueError(f"文件夹不存在: {image_folder}")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取图像文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = sorted([f for f in os.listdir(image_folder) 
                            if f.lower().endswith(image_extensions)])
        
        if len(image_files) < 2:
            raise ValueError("文件夹中的图像文件少于2张")
        
        print(f"在文件夹中找到 {len(image_files)} 张图像")
        
        # 如果使用保存的变换，先加载
        if use_saved_transform and transform_file and os.path.exists(transform_file):
            self.load_transform_from_file(transform_file)
            print("已加载保存的变换参数")
        
        successful_files = []
        
        # 按对处理图像
        for i in range(0, len(image_files)-1, 2):
            if i+1 < len(image_files):
                imageA_path = os.path.join(image_folder, image_files[i])
                imageB_path = os.path.join(image_folder, image_files[i+1])
                
                print(f"\n正在处理第 {i//2+1} 对: {image_files[i]} 和 {image_files[i+1]}")
                
                imageA = cv2.imread(imageA_path)
                imageB = cv2.imread(imageB_path)
                
                if imageA is not None and imageB is not None:
                    try:
                        output_filename = f"stitched_pair_{i//2+1:03d}_{image_files[i].split('.')[0]}_{image_files[i+1].split('.')[0]}.jpg"
                        output_path = os.path.join(output_folder, output_filename)
                        
                        if use_saved_transform and self.saved_transform is not None:
                            # 使用保存的变换
                            result = self.stitch_with_saved_transform(
                                [imageB, imageA],
                                save_result=True,
                                result_save_path=output_path
                            )
                        else:
                            # 使用特征点匹配
                            result = self.stitch_with_feature_matching(
                                [imageB, imageA],
                                save_result=True,
                                result_save_path=output_path
                            )
                        
                        if result is not None:
                            successful_files.append(output_path)
                            print(f"拼接成功: {output_filename}")
                        else:
                            print(f"拼接失败: {image_files[i]} 和 {image_files[i+1]}")
                            
                    except Exception as e:
                        print(f"处理错误: {e}")
                else:
                    print(f"无法读取图像: {image_files[i]} 或 {image_files[i+1]}")
        
        print(f"\n成对拼接完成: {len(successful_files)}/{(len(image_files)-1)//2 + 1} 对图像成功拼接")
        return successful_files
    
    def create_panorama_grid(self, image_folder: str, output_path: str,
                           grid_cols: int = 2, resize_factor: float = 0.5) -> np.ndarray:
        """
        将文件夹中的图像排列成网格形式的全景图
        
        Args:
            image_folder: 图像文件夹路径
            output_path: 输出路径
            grid_cols: 网格列数
            resize_factor: 图像缩放因子
            
        Returns:
            网格全景图
        """
        if not os.path.exists(image_folder):
            raise ValueError(f"文件夹不存在: {image_folder}")
        
        # 获取所有图像文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = sorted([f for f in os.listdir(image_folder) 
                            if f.lower().endswith(image_extensions)])
        
        if len(image_files) == 0:
            raise ValueError("文件夹中没有图像文件")
        
        print(f"创建 {len(image_files)} 张图像的网格全景图")
        
        # 读取并调整所有图像大小
        images = []
        max_height = 0
        max_width = 0
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                # 调整图像大小
                if resize_factor != 1.0:
                    new_width = int(img.shape[1] * resize_factor)
                    new_height = int(img.shape[0] * resize_factor)
                    img = cv2.resize(img, (new_width, new_height))
                
                images.append(img)
                max_height = max(max_height, img.shape[0])
                max_width = max(max_width, img.shape[1])
                print(f"已加载: {img_file} - 调整后尺寸: {img.shape}")
        
        if not images:
            raise ValueError("没有成功加载任何图像")
        
        # 计算网格行数
        grid_rows = (len(images) + grid_cols - 1) // grid_cols
        
        print(f"创建 {grid_rows}x{grid_cols} 网格")
        
        # 创建统一尺寸的图像
        uniform_images = []
        for img in images:
            # 创建黑色背景
            uniform_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            
            # 居中放置原图像
            y_offset = (max_height - img.shape[0]) // 2
            x_offset = (max_width - img.shape[1]) // 2
            uniform_img[y_offset:y_offset+img.shape[0], 
                       x_offset:x_offset+img.shape[1]] = img
            
            uniform_images.append(uniform_img)
        
        # 填充空白图像以完整网格
        while len(uniform_images) < grid_rows * grid_cols:
            blank_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            uniform_images.append(blank_img)
        
        # 创建网格
        rows = []
        for r in range(grid_rows):
            row_images = uniform_images[r * grid_cols:(r + 1) * grid_cols]
            row = np.hstack(row_images)
            rows.append(row)
        
        # 垂直堆叠所有行
        grid_panorama = np.vstack(rows)
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            success = cv2.imwrite(output_path, grid_panorama)
            if success:
                print(f"网格全景图已保存到: {output_path}")
            else:
                print(f"保存失败: {output_path}")
        
        return grid_panorama
    
    def analyze_folder_images(self, image_folder: str) -> dict:
        """
        分析文件夹中图像的基本信息
        
        Args:
            image_folder: 图像文件夹路径
            
        Returns:
            包含图像信息的字典
        """
        if not os.path.exists(image_folder):
            raise ValueError(f"文件夹不存在: {image_folder}")
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        # image_files = sorted([f for f in os.listdir(image_folder) 
        #                     if f.lower().endswith(image_extensions)])
        filelist = os.listdir(image_folder)
        image_files = sorted(
            [f for f in filelist if f.lower().endswith(image_extensions)],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        
        analysis = {
            'total_images': len(image_files),
            'image_files': image_files,
            'image_info': [],
            'size_stats': {'widths': [], 'heights': [], 'file_sizes': []}
        }
        
        print(f"分析文件夹: {image_folder}")
        print(f"找到 {len(image_files)} 张图像")
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            
            # 获取文件大小
            file_size = os.path.getsize(img_path)
            
            # 读取图像获取尺寸
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                
                img_info = {
                    'filename': img_file,
                    'width': width,
                    'height': height,
                    'channels': img.shape[2] if len(img.shape) > 2 else 1,
                    'file_size_mb': file_size / (1024 * 1024)
                }
                
                analysis['image_info'].append(img_info)
                analysis['size_stats']['widths'].append(width)
                analysis['size_stats']['heights'].append(height)
                analysis['size_stats']['file_sizes'].append(file_size / (1024 * 1024))
                
                print(f"  {img_file}: {width}x{height}, {img_info['file_size_mb']:.2f}MB")
            else:
                print(f"  警告: 无法读取 {img_file}")
        
        # 计算统计信息
        if analysis['size_stats']['widths']:
            widths = analysis['size_stats']['widths']
            heights = analysis['size_stats']['heights']
            file_sizes = analysis['size_stats']['file_sizes']
            
            analysis['statistics'] = {
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'avg_file_size_mb': np.mean(file_sizes),
                'min_width': min(widths),
                'max_width': max(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'total_size_mb': sum(file_sizes)
            }
            
            print(f"\n统计信息:")
            print(f"  平均尺寸: {analysis['statistics']['avg_width']:.0f}x{analysis['statistics']['avg_height']:.0f}")
            print(f"  尺寸范围: {analysis['statistics']['min_width']}x{analysis['statistics']['min_height']} 到 {analysis['statistics']['max_width']}x{analysis['statistics']['max_height']}")
            print(f"  平均文件大小: {analysis['statistics']['avg_file_size_mb']:.2f}MB")
            print(f"  总文件大小: {analysis['statistics']['total_size_mb']:.2f}MB")
        
        return analysis
    
    # 保持原有的核心方法
    def stitch_with_feature_matching(self, images, ratio=0.75, reprojThresh=4.0, 
                                   showMatches=False, save_transform=True, 
                                   transform_save_path=None, save_result=False,
                                   result_save_path=None, matches_save_path=None):
        """使用特征点匹配进行拼接（保持原有实现）"""
        if len(images) != 2:
            raise ValueError("需要输入两张图像")
            
        (imageB, imageA) = images
        
        if imageA is None or imageB is None:
            raise ValueError("输入图像为空")
        
        # 特征点检测和匹配
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        
        if featuresA is None or featuresB is None:
            print("警告: 无法在图像中检测到特征点")
            return None
        
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        
        if M is None:
            print("警告: 未找到足够的匹配点")
            return None
            
        (matches, H, status) = M
        print(f"找到 {len(matches)} 个匹配点, {np.sum(status)} 个内点")
        
        # 计算变换参数
        transform_params = self.calculate_transform_params(imageA, imageB, H)
        
        # 保存变换参数
        if save_transform:
            self.saved_transform = H.copy()
            self.transform_params = transform_params.copy()
        
        # 执行拼接
        result = self.apply_transform_and_blend(imageA, imageB, H, transform_params)
        
        # 保存结果
        if save_result and result_save_path:
            os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
            cv2.imwrite(result_save_path, result)
            print(f"拼接结果已保存到: {result_save_path}")
        
        return result
    
    def stitch_with_saved_transform(self, images, transform_matrix=None, 
                                  transform_params=None, blend_mode='linear',
                                  save_result=False, result_save_path=None):
        """使用保存的变换矩阵进行拼接（保持原有实现）"""
        if len(images) != 2:
            raise ValueError("需要输入两张图像")
            
        (imageB, imageA) = images
        
        H = transform_matrix if transform_matrix is not None else self.saved_transform
        params = transform_params if transform_params is not None else self.transform_params
        
        if H is None:
            raise ValueError("没有可用的变换矩阵")
        
        if params is None:
            params = self.calculate_transform_params(imageA, imageB, H)
        
        result = self.apply_transform_and_blend(imageA, imageB, H, params, blend_mode)
        
        if save_result and result_save_path:
            os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
            cv2.imwrite(result_save_path, result)
        
        return result
    
    # 其他辅助方法保持不变...
    def detectAndDescribe(self, image):
        """检测关键点和计算描述符"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (kps, features) = self.feature_detector.detectAndCompute(gray, None)
        
        if kps is None or len(kps) == 0:
            return (None, None)
        
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        """匹配关键点"""
        if self.feature_detector_type == 'ORB':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        try:
            rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)
        except cv2.error:
            return None
        
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        
        if len(matches) < 4:
            return None
        
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        
        try:
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        except cv2.error:
            return None
        
        return (matches, H, status) if H is not None else None
    
    def calculate_transform_params(self, imageA, imageB, H):
        """计算变换参数"""
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        
        corners = np.float32([[0, 0], [wA, 0], [wA, hA], [0, hA]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        
        all_corners = np.concatenate([
            warped_corners, 
            np.float32([[0, 0], [wB, 0], [wB, hB], [0, hB]]).reshape(-1, 1, 2)
        ])
        
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # 转换为Python原生类型
        translation_dist = [int(-x_min), int(-y_min)]
        result_width = int(x_max - x_min)
        result_height = int(y_max - y_min)
        
        H_translation = np.array([
            [1, 0, translation_dist[0]], 
            [0, 1, translation_dist[1]], 
            [0, 0, 1]
        ])
        
        return {
            'translation_dist': translation_dist,
            'result_width': result_width,
            'result_height': result_height,
            'H_translation': H_translation,
            'original_imageA_size': [int(hA), int(wA)],  # 转换为Python int
            'original_imageB_size': [int(hB), int(wB)]   # 转换为Python int
        }
    
    def apply_transform_and_blend(self, imageA, imageB, H, params, blend_mode='linear'):
        """应用变换和融合"""
        H_combined = params['H_translation'].dot(H)
        warped_imageA = cv2.warpPerspective(imageA, H_combined, 
                                          (params['result_width'], params['result_height']))
        
        result = warped_imageA.copy()
        translation_dist = params['translation_dist']
        hB, wB = params['original_imageB_size']
        
        result[translation_dist[1]:translation_dist[1]+hB, 
               translation_dist[0]:translation_dist[0]+wB] = imageB
        
        if blend_mode == 'linear':
            result = self.linear_blend(result, warped_imageA, imageB, params)
        
        return result
    
    def linear_blend(self, result, warped_imageA, imageB, params):
        """线性融合重叠区域"""
        mask_A = (warped_imageA > 0).astype(np.float32)
        mask_B = np.zeros_like(mask_A)
        
        translation_dist = params['translation_dist']
        hB, wB = params['original_imageB_size']
        
        mask_B[translation_dist[1]:translation_dist[1]+hB, 
               translation_dist[0]:translation_dist[0]+wB] = 1.0
        
        overlap = mask_A * mask_B
        
        if np.sum(overlap) > 0:
            overlap_region = overlap[:, :, 0] if len(overlap.shape) == 3 else overlap
            
            for y in range(result.shape[0]):
                for x in range(result.shape[1]):
                    if overlap_region[y, x] > 0:
                        imageB_left = translation_dist[0]
                        imageB_right = translation_dist[0] + wB
                        
                        if imageB_left <= x <= imageB_right:
                            weight = (x - imageB_left) / wB
                            weight = max(0, min(1, weight))
                            
                            result[y, x] = (weight * warped_imageA[y, x] + 
                                          (1 - weight) * result[y, x])
        
        return result
    
    def load_transform_from_file(self, filepath):
        """从文件加载变换参数"""
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            H = np.array(save_data['homography_matrix'], dtype=np.float64)
            params = save_data['transform_params']
            params['H_translation'] = np.array(params['H_translation'], dtype=np.float64)
            
            self.saved_transform = H
            self.transform_params = params
            
            return H, params
        except Exception as e:
            raise Exception(f"加载变换参数失败: {e}")
        

    
    def apply_sequential_transforms_to_folder(self, source_folder: str, 
                                            transforms_file: str,
                                            output_folder: str,
                                            base_image_path: str = None) -> List[str]:
        """
        使用保存的顺序变换参数将另一个文件夹中的图像进行拼接
        
        Args:
            source_folder: 源图像文件夹路径
            transforms_file: 变换参数文件路径
            output_folder: 输出文件夹路径
            base_image_path: 基础图像路径（如果为None，使用第一张图像）
            
        Returns:
            处理成功的文件列表
        """
        if not os.path.exists(source_folder):
            raise ValueError(f"源文件夹不存在: {source_folder}")
        
        if not os.path.exists(transforms_file):
            raise ValueError(f"变换参数文件不存在: {transforms_file}")
        
        # 加载变换参数
        with open(transforms_file, 'r', encoding='utf-8') as f:
            transform_data = json.load(f)
        
        transforms = transform_data['transforms']
        metadata = transform_data['metadata']
        
        print(f"加载变换参数:")
        print(f"  创建时间: {metadata['creation_time']}")
        print(f"  基础图像: {metadata['base_image']}")
        print(f"  变换步骤数: {len(transforms)}")
        
        # 获取源文件夹中的图像文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        filelist = os.listdir(source_folder)
        source_images = sorted(
            [f for f in filelist if f.lower().endswith(image_extensions)],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        
        if len(source_images) < len(transforms) + 1:
            print(f"警告: 源文件夹中图像数量({len(source_images)})少于变换步骤数({len(transforms) + 1})")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 读取基础图像
        if base_image_path:
            base_image = cv2.imread(base_image_path)
        else:
            base_image = cv2.imread(os.path.join(source_folder, source_images[0]))
        
        if base_image is None:
            raise ValueError("无法读取基础图像")
        
        print(f"基础图像尺寸: {base_image.shape}")
        current_result = base_image.copy()
        successful_files = []
        
        # 逐步应用变换
        for i, transform_info in enumerate(transforms):
            step = transform_info['step']
            source_image_idx = step  # step对应的是第几张图像的索引
            
            if source_image_idx < len(source_images):
                source_image_path = os.path.join(source_folder, source_images[source_image_idx])
                source_image = cv2.imread(source_image_path)
                
                if source_image is None:
                    print(f"警告: 无法读取源图像 {source_images[source_image_idx]}")
                    continue
                
                print(f"\n步骤 {step}: 应用变换到 {source_images[source_image_idx]}")
                
                # 重建变换矩阵和参数
                H = np.array(transform_info['homography_matrix'], dtype=np.float64)
                params = {
                    'translation_dist': transform_info['transform_params']['translation_dist'],
                    'result_width': transform_info['transform_params']['result_width'],
                    'result_height': transform_info['transform_params']['result_height'],
                    'H_translation': np.array(transform_info['transform_params']['H_translation'], dtype=np.float64),
                    'original_imageA_size': transform_info['transform_params']['original_imageA_size'],
                    'original_imageB_size': transform_info['transform_params']['original_imageB_size']
                }
                
                # 应用变换
                try:
                    result = self.apply_transform_and_blend(current_result, source_image, H, params)
                    
                    if result is not None:
                        current_result = result
                        
                        # 保存中间结果
                        intermediate_filename = f"step_{step:02d}_{source_images[source_image_idx]}"
                        intermediate_path = os.path.join(output_folder, intermediate_filename)
                        cv2.imwrite(intermediate_path, result)
                        successful_files.append(intermediate_path)
                        
                        print(f"  成功应用变换，结果尺寸: {result.shape}")
                    else:
                        print(f"  变换应用失败")
                        
                except Exception as e:
                    print(f"  应用变换时出错: {e}")
            else:
                print(f"警告: 源文件夹中没有对应的第 {source_image_idx + 1} 张图像")
        
        # 保存最终结果
        final_output_path = os.path.join(output_folder, "final_sequential_result.jpg")
        cv2.imwrite(final_output_path, current_result)
        successful_files.append(final_output_path)
        
        print(f"\n顺序变换应用完成:")
        print(f"  成功处理步骤: {len([f for f in successful_files if 'step_' in f])}")
        print(f"  最终结果: {final_output_path}")
        
        return successful_files

    def load_sequential_transforms(self, transforms_file: str) -> dict:
        """
        加载顺序变换参数文件
        
        Args:
            transforms_file: 变换参数文件路径
            
        Returns:
            变换参数数据
        """
        if not os.path.exists(transforms_file):
            raise ValueError(f"变换参数文件不存在: {transforms_file}")
        
        with open(transforms_file, 'r', encoding='utf-8') as f:
            transform_data = json.load(f)
        
        self.sequential_transforms = transform_data['transforms']
        
        return transform_data

    def save_sequential_transforms(self, save_path: str, metadata: dict = None):
        """
        手动保存当前的顺序变换参数
        
        Args:
            save_path: 保存路径
            metadata: 额外的元数据
        """
        if not self.sequential_transforms:
            raise ValueError("没有可保存的变换参数")
        
        transform_data = {
            'metadata': {
                'creation_time': datetime.datetime.now().isoformat(),
                'feature_detector': self.feature_detector_type,
                'total_steps': len(self.sequential_transforms),
                **(metadata or {})
            },
            'transforms': self.sequential_transforms
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(transform_data, f, indent=2, ensure_ascii=False)
        
        print(f"变换参数已保存到: {save_path}")
        

    def rectify_result(self, image, margin=50):
        """
        校正拼接结果为矩形
        
        Args:
            image: 输入图像
            margin: 边缘裁剪margin
            
        Returns:
            校正后的矩形图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 创建掩码，找到非黑色区域
        mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        
        # 形态学操作
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 找到轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # 获取最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.float32(box)
        
        # 排序角点：左上、右上、右下、左下
        center = np.mean(box, axis=0)
        
        def sort_key(point):
            return (point[0] - center[0]) + (point[1] - center[1])
        
        sorted_points = sorted(box, key=sort_key)
        
        # 重新排列为正确顺序
        top_left = sorted_points[0]
        bottom_right = sorted_points[3]
        
        remaining = [p for p in box if not np.array_equal(p, top_left) and not np.array_equal(p, bottom_right)]
        top_right = max(remaining, key=lambda p: (p[0] - center[0]) - (p[1] - center[1]))
        bottom_left = min(remaining, key=lambda p: (p[0] - center[0]) - (p[1] - center[1]))
        
        src_points = np.float32([top_left, top_right, bottom_right, bottom_left])
        
        # 计算输出尺寸
        width_top = np.linalg.norm(src_points[0] - src_points[1])
        width_bottom = np.linalg.norm(src_points[3] - src_points[2])
        height_left = np.linalg.norm(src_points[0] - src_points[3])
        height_right = np.linalg.norm(src_points[1] - src_points[2])
        
        max_width = int(max(width_top, width_bottom))
        max_height = int(max(height_left, height_right))
        
        # 目标角点
        dst_points = np.float32([
            [0, 0],
            [max_width, 0],
            [max_width, max_height],
            [0, max_height]
        ])
        
        # 透视变换
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        rectified = cv2.warpPerspective(image, perspective_matrix, (max_width, max_height))
        
        # 边缘裁剪
        if margin > 0 and margin < min(max_width, max_height) // 4:
            rectified = rectified[margin:max_height-margin, margin:max_width-margin]
        
        return rectified
    
    def stitch_multiple_images_sequential_with_rectify(self, image_folder: str, output_path: str = None,
                                                 save_intermediate: bool = False,
                                                 intermediate_folder: str = None,
                                                 save_transforms: bool = True,
                                                 transforms_save_path: str = None,
                                                 rectify_result: bool = True,
                                                 rectify_margin: int = 50) -> np.ndarray:
        """
        顺序拼接并校正结果
        
        Args:
            rectify_result: 是否校正最终结果为矩形
            rectify_margin: 校正时的边缘裁剪像素数
            其他参数同原方法
        """
        # 调用原来的拼接方法
        result = self.stitch_multiple_images_sequential(
            image_folder=image_folder,
            output_path=None,  # 先不保存，等校正后再保存
            save_intermediate=save_intermediate,
            intermediate_folder=intermediate_folder,
            save_transforms=save_transforms,
            transforms_save_path=transforms_save_path
        )
        
        if result is None:
            return None
        
        # 如果需要校正
        if rectify_result:
            print("\n--- 校正拼接结果为矩形 ---")
            print(f"原始拼接结果尺寸: {result.shape}")
            
            rectified_result = self.rectify_result(result, margin=rectify_margin)
            
            if rectified_result is not None:
                print(f"校正后尺寸: {rectified_result.shape}")
                result = rectified_result
            else:
                print("校正失败，使用原始结果")
        
        # 保存最终结果
        if output_path:
            if os.path.dirname(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            success = cv2.imwrite(output_path, result)
            if success:
                print(f"\n最终结果已保存到: {output_path}")
            else:
                print(f"保存失败: {output_path}")
        
        return result