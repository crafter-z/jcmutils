"""此模块负责数据集的生成
数据会被存放在单独的dataset文件夹里
"""
import random
import os
import numpy as np
import cv2

from .logger import logger


class datagen:
    """在使用数据集生成器之前，需要先对其进行初始化"""

    def __init__(self):
        logger.debug("datagen inited,no error reported")
        # 随机初始化
        random.seed()

    def export_defect_datas(
        self, template_image, target_image, periodic_info, signal_level, defect_class
    ):
        """导出一个有缺陷图像所对应的各种信息
        :param  template_image: 无缺陷的模板图像，是图像而不是图像的路径
        :param  target_image: 有缺陷的目标图像
        :param  periodic_info: 周期性信息，第一位是多少列像素一个周期，第二位是多少行像素一个周期
        :param  signal_level: 缺陷信号强度，作为论文中的ε来判断阈值
        :param  defect_class: 该缺陷是哪一类缺陷
        :returns -> List:返回值是一个列表
                第一项是抠出来的缺陷所在的矩形区域的图像
                第二项是periodic_info
                第三项是缺陷种类
                第四项是缺陷在原图中的对应(x,y,w,h)，其中x，y是左上角坐标，w,h是宽高
                第五项是bool类型，代表该图所对应缺陷是否被划入训练或验证集
        """
        datas = self.__process_image(
            target_image,
            template_image,
            signal_level,
            periodic_info,
            defect_class,
            5,
            3,
        )
        return datas

    def export_dataset(
        self,
        list_of_datas,
        template_image,
        target_shape,
        source_density,
        target_density,
        target_directory,
        periodic_info,
        enhance_info,
        defect_num_one_image,
        min_required_num,
    ):
        """将一个由export_defect_datas中返回的datas组成的列表处理为数据集并保存
        :param  list_of_datas: 从export_defect_datas函数中生成的datas组成的list
        :param  template_image: 无缺陷的样本图像
        :param  target_shape: 导出目标图像的分辨率，缩放后的
        :param  source_density: 原图像一个像素代表多少个纳米
        :param  target_density: 导出目标的一个像素代表target_density*target_density纳米
        :param  target_directory: 输出数据的目标文件夹名
        :param  periodic_info: 周期性信息，第一位是多少列像素一个周期，第二位是多少行像素一个周期
        :param  enhance_info: 包含可用数据增强（含噪声信息）的字典
        :param  defect_num_one_image: 生成的每张图像中要包含多少个缺陷
        :param  min_required_num: 至少需要生成多少张图像

        """
        # 路径预处理
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # 准备必须的数据
        periodic_x = periodic_info[0]
        periodic_y = periodic_info[1]

        # 创建定长数组,存储一个缺陷是否已被生成的状态
        defect_num = len(list_of_datas)
        defect_count = [0] * defect_num

        # 获得在缩放之前的仿真图像的尺寸
        temp_shape = target_shape.copy()
        temp_shape[0] = int(target_shape[0] * target_density / source_density)
        temp_shape[1] = int(target_shape[1] * target_density / source_density)
        template_reformed = cv2.copyMakeBorder(
            template_image[0 : 2 * periodic_info[1], 0 : 2 * periodic_info[0]],
            0,
            temp_shape[0] - 2 * periodic_info[1],
            0,
            temp_shape[1] - 2 * periodic_info[0],
            cv2.BORDER_WRAP,
        )

        # 获取文件夹中文件名的最大数值，保存的每张图像的名字应该按顺序增加
        path_list = os.listdir(target_directory)
        if len(path_list) == 0:
            image_tag = 0
        else:
            path_list.sort(
                key=lambda x: int(x.split(".")[0])
            )  # 对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
            temp_name, _ = os.path.splitext(path_list[-1])
            image_tag = int(temp_name)

        # 开始处理每张图像,迭代终止条件是所有缺陷均被包含且图像数量不小于min_required_num
        while (0 in defect_count) or (image_tag < min_required_num):
            # 深拷贝图像
            current_image = template_reformed.copy()
            image_tag += 1
            picked_lists = []  # 存储当前图像中已被添加的缺陷
            defect_text_list = []  # 存储被保存进数据集标注文件中的信息

            # 计次循环
            for _ in range(defect_num_one_image):
                picked_tag = random.randint(0, defect_num - 1)
                defect_count[picked_tag] += 1
                picked_datas = list_of_datas[picked_tag]

                while True:
                    # 在允许的范围内进行随机移动,表示在横向第几个周期，和纵向第几个周期
                    rand_defectpos = [
                        random.randint(
                            0,
                            int((temp_shape[1] - picked_datas[3][1]) / periodic_y) - 3,
                        ),
                        random.randint(
                            0,
                            int((temp_shape[0] - picked_datas[3][1]) / periodic_x) - 3,
                        ),
                    ]
                    if len(picked_lists) == 0:
                        break
                    for pos in picked_lists:
                        if (pos[0] - rand_defectpos[0]) ** 2 + (
                            pos[1] - rand_defectpos[1]
                        ) ** 2 < 9:
                            continue
                    break

                picked_lists.append(rand_defectpos)
                # 将缺陷图像粘贴到无缺陷图像的随机位置
                base_y = picked_datas[3][1] + periodic_y * rand_defectpos[0]
                base_x = picked_datas[3][0] + periodic_x * rand_defectpos[1]
                current_image[
                    base_y : base_y + picked_datas[3][3],
                    base_x : base_x + picked_datas[3][2],
                ] = picked_datas[0][0 : picked_datas[3][3], 0 : picked_datas[3][2]]

                # 计算yolo格式数据集涉及到的信息
                xpos = (base_x + picked_datas[3][2] / 2) / temp_shape[1]
                ypos = (base_y + picked_datas[3][3] / 2) / temp_shape[0]
                width = picked_datas[3][2] / temp_shape[1]
                height = picked_datas[3][3] / temp_shape[0]

                defect_text_list.append(
                    f"{picked_datas[2]} {xpos} {ypos} {width} {height}\n"
                )

            label_name = os.path.join(target_directory, f"{image_tag}.txt")
            file_name = os.path.join(target_directory, f"{image_tag}.jpg")

            # 图像处理开始------------------------

            # 缩放
            scale_factor = source_density * 1.0 / target_density
            output_image = cv2.resize(
                current_image,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_LINEAR,
            )
            # 接下来向图像中添加噪声
            if "noise_level" in enhance_info:
                noise_level = enhance_info["noise_level"]
                # 高斯噪声参数
                mean = 0
                # 根据峰值信噪比计算高斯噪声的标准差
                sigma = np.sqrt(255**2 / (10 ** (noise_level / 10)))
                image_shape = (output_image.shape[0], output_image.shape[1])
                gauss = np.random.normal(mean, sigma, image_shape)
                output_image = np.clip(output_image + gauss, 0, 255)

            # 图像处理结束--------------------------

            # 保存图像与数据集文件
            with open(label_name, "w",encoding="utf-8") as f:
                for text in defect_text_list:
                    f.write(text)
            cv2.imwrite(file_name, output_image)
            logger.info(f"target image {image_tag} saved completed!")

    def __process_image(
        self,
        defect_img,
        template_img,
        signal_level,
        periodic_info,
        defect_class,
        smooth_length=5,
        extend_length=3,
    ):
        """处理图像
        :param  smooth_length: 获取缺陷区域后，要在缺陷外一段区域进行一下平滑处理
        :param  extend_length: 返回缺陷区域的时候，在缺陷外部再延伸一点区域返回,避免算法硬截断
        """
        diff_img = defect_img.astype(np.float32) - template_img.astype(np.float32)
        image_shape = template_img.shape
        # diff_img = (diff_img + 125)
        # diff_img = np.clip(diff_img, 0, 255).astype(np.uint8)

        gradX = cv2.Sobel(diff_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(diff_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

        gradient = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
        gradient = cv2.convertScaleAbs(gradient)
        defect_lowborder = np.max(gradient) * signal_level
        # blurred = cv2.blur(gradient, (5, 5),borderType=cv2.BORDER_REFLECT)
        (_, thresh) = cv2.threshold(gradient, defect_lowborder, 255, cv2.THRESH_BINARY)

        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            kernel1,
            iterations=2,
            borderType=cv2.BORDER_ISOLATED,
        )
        thresh = cv2.dilate(thresh, kernel1, iterations=2)
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel2,
            iterations=2,
            borderType=cv2.BORDER_ISOLATED,
        )
        thresh = cv2.dilate(thresh, kernel1, iterations=2)

        # 找距离图像中心点最近的一个封闭区域
        (cnts, _) = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_dist = -1
        c = cnts[0]
        for conners in cnts:
            x, y, w, h = cv2.boundingRect(conners)
            rect_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            distances = []
            for k in range(4):
                # 获取当前边的起点和终点
                p1 = rect_points[k]
                p2 = rect_points[(k + 1) % 4]

                # 计算点到当前边的距离
                distance = cv2.pointPolygonTest(
                    np.array([p1, p2], np.int32),
                    (image_shape[1] / 2, image_shape[0] / 2),
                    True,
                )
                distances.append(abs(distance))
            dist = min(distances)
            if dist < min_dist or min_dist == -1:
                min_dist = dist
                c = conners

        # compute the rotated bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.2 * periodic_info[0] or w > 1.5 * periodic_info[0]:
            raise Exception(f"缺陷提取出现错误，当前宽度为{w}")
        if h < 0.2 * periodic_info[1] or w > 1.5 * periodic_info[1]:
            raise Exception(f"缺陷提取出现错误，当前高度为{h}")

        # 延伸扩展边界，避免强截断
        x -= extend_length
        y -= extend_length
        w += extend_length * 2
        h += extend_length * 2

        # img=cv2.rectangle(defect_img,(x,y),(x+w,y+h),(0,255,0),2)
        # 根据左上角坐标和长宽计算矩形的四个角点坐标
        rect_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        # 开始扩展拼接缺陷图像
        outer_points = [
            (x - smooth_length, y - smooth_length),
            (x + w + smooth_length, y - smooth_length),
            (x + w + smooth_length, y + h + smooth_length),
            (x - smooth_length, y + h + smooth_length),
        ]

        process_img = template_img.copy()
        process_img[y : y + h, x : x + w] = defect_img[y : y + h, x : x + w]

        x_lower_border = max(0, x - smooth_length)
        x_upper_border = min(image_shape[1] - 1, x + w + smooth_length - 1)
        y_lower_border = max(0, y - smooth_length)
        y_upper_border = min(image_shape[0] - 1, y + h + smooth_length - 1)

        for i in range(x_lower_border, x_upper_border):
            for j in range(y_lower_border, y_upper_border):
                if not (
                    np.abs(i - x - w / 2 + 0.5) < w / 2
                    and np.abs(j - y - h / 2 + 0.5) < h / 2
                ):
                    # 计算点到矩形边界的距离
                    distances = []
                    distances2 = []
                    for k in range(4):
                        # 获取当前边的起点和终点
                        p1 = rect_points[k]
                        p2 = rect_points[(k + 1) % 4]
                        p11 = outer_points[k]
                        p22 = outer_points[(k + 1) % 4]

                        # 计算点到当前边的距离
                        distance = cv2.pointPolygonTest(
                            np.array([p1, p2], np.int32), (i, j), True
                        )
                        distances.append(abs(distance))
                        distance2 = cv2.pointPolygonTest(
                            np.array([p11, p22], np.int32), (i, j), True
                        )
                        distances2.append(abs(distance2))

                    # 获取最短距离
                    min_distance = min(distances)
                    min_distance2 = min(distances2)
                    # output_img[j,i] = 255
                    process_img[j, i] += (
                        diff_img[j, i]
                        * (min_distance2)
                        / (min_distance + min_distance2)
                    )
        output_img = process_img[
            y_lower_border:y_upper_border, x_lower_border:x_upper_border
        ]

        p = random.random()
        return (
            output_img,
            periodic_info,
            defect_class,
            (
                x_lower_border,
                y_lower_border,
                x_upper_border - x_lower_border,
                y_upper_border - y_lower_border,
            ),
            p < 0.9,
        )
