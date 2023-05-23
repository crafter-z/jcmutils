from .logger import logger
import numpy as np
import os
import jcmwave
import cv2
import yaml


class datagen:
    def __init__(self, jcmp_path, database_path, keys):
        # 初始化成员变量
        self.jcmp_path = jcmp_path
        self.keys = keys
        if os.path.isabs(database_path):
            abs_resultbag_dir = database_path
        else:
            abs_resultbag_dir = os.path.join(os.getcwd(), database_path)
        if not os.path.exists(os.path.dirname(database_path)):
            raise Exception("exporting dataset but resultbag dosen't exist")
        self.resultbag = jcmwave.Resultbag(abs_resultbag_dir)
        logger.debug("datagen inited,no error reported")
        logger.debug(
            f"jcmp_path is {jcmp_path},database_path is {abs_resultbag_dir}")

    def export_dataset_one(self, num_of_result, source_density, target_density,target_filename,phi0, vmax, is_light_intense=True, is_symmetry=False):
        # 路径预处理
        if not os.path.exists(os.path.dirname(target_filename)):
            os.makedirs(os.path.dirname(target_filename))

        # 提取无缺陷图像
        ## 先确定total_result的形状
        temp_result = self.resultbag.get_result(self.keys[0])
        field = (temp_result[num_of_result]['field'][0].conj() *
                 temp_result[num_of_result]['field'][0]).sum(axis=2).real
        total_results = np.zeros(field.shape)
        logger.debug(f"total_result shape defined as {total_results.shape}")

        ## 开始逐个提取结果
        for key in self.keys:
            result = self.resultbag.get_result(key)
            field = (result[num_of_result]['field'][0].conj() *
                     result[num_of_result]['field'][0]).sum(axis=2).real
            if is_light_intense:
                field = np.power(field, 2)
            total_results += field
            if is_symmetry and not (key['thetaphi'][0] == 0 and key['thetaphi'][1] == 0):
                field = np.rot90(field, 2)
                total_results += field
                logger.debug("key was rotated for symmetry")
        
        # 合并最终结果
        vmaxa = np.max(total_results) if vmax is None else vmax
        afield = (total_results/ vmaxa)*235
        afield = np.rot90(afield)

        label_name = target_filename + ".txt"
        with open(label_name,"w") as f:
            f.write("")

        # 保存超分辨（原图）
        cv2.imwrite(target_filename + "_origin.jpg",afield)

        # 通过每个像素点代表的实际物理尺寸来计算缩放比比例
        scale_factor =source_density*1.0/target_density
        # 缩放电场/光强场到对应的大小
        scaled_field = cv2.resize(output_image, None, fx=scale_factor,# type: ignore
                                  fy=scale_factor, interpolation=cv2.INTER_LINEAR)  

        # 绘图
        logger.debug(f"printing max value of results:{np.max(total_results)}")
        cv2.imwrite(target_filename + ".jpg",scaled_field)
        logger.info("all target image saved completed!")



    def export_dataset(self, num_of_result, source_density, target_density,target_filename,phi0,defect_size, vmax, is_light_intense=True, is_symmetry=False):
        # 路径预处理
        if not os.path.exists(os.path.dirname(target_filename)):
            os.makedirs(os.path.dirname(target_filename))
        yamlpath =os.path.join(os.path.dirname(self.jcmp_path),"properties.yaml")
        
        # 解析YAML，准备必须的数据
        with open(yamlpath) as f:
            data = yaml.load(f,Loader=yaml.FullLoader)
        lattice_vector_length = data['latticeVector']['latticeVectorLength']
        lattice_angle = data['latticeVector']['latticeAngle']
        center_pos = data['centerPos']
        shift = data['shift']
        nodefect_phi0_0 = data['nodefect']['phi0-0']
        nodefect_phi0_90 = data['nodefect']['phi0-90']
        origin_size = data['originSize'] 
        
        # 获取模板图像
        if phi0 == 90:
            template_path = nodefect_phi0_90
        else:
            template_path = nodefect_phi0_0
        template_image = cv2.imread(template_path,cv2.IMREAD_GRAYSCALE)
        origin_image_size = template_image.shape
        
        # 确定缺陷类别
        defect_class = 2
        if "instruction" in target_filename:
            defect_class = 0
        elif "particle" in target_filename:
            defect_class = 1

        # 提取周期性缺陷图像
        ## 先确定total_result的形状
        temp_result = self.resultbag.get_result(self.keys[0])
        field = (temp_result[num_of_result]['field'][0].conj() *
                 temp_result[num_of_result]['field'][0]).sum(axis=2).real
        total_results = np.zeros(field.shape)
        logger.debug(f"total_result shape defined as {total_results.shape}")

        ## 开始逐个提取结果
        for key in self.keys:
            result = self.resultbag.get_result(key)
            field = (result[num_of_result]['field'][0].conj() *
                     result[num_of_result]['field'][0]).sum(axis=2).real
            if is_light_intense:
                field = np.power(field, 2)
            total_results += field
            if is_symmetry and not (key['thetaphi'][0] == 0 and key['thetaphi'][1] == 0):
                field = np.rot90(field, 2)
                total_results += field
                logger.debug("key was rotated for symmetry")
        
        # 合并最终结果
        vmaxa = np.max(total_results) if vmax is None else vmax
        afield = (total_results/ vmaxa)*235
        afield = np.rot90(afield)

        # # 确定缺陷在原始图像中的位置
        # xpos = (self.keys[0]['defectpos'][0] - origin_size['x'][0])* 1.0/( origin_size['x'][1] - origin_size['x'][0]) * origin_image_size[0]
        # ypos = origin_image_size[1]-((self.keys[0]['defectpos'][1] - origin_size['y'][0]) * 1.0/(origin_size['y'][1] - origin_size['y'][0]) * origin_image_size[1])
        # shift_pix = defect_size*1.0/source_density
        # roi = [xpos - shift_pix , xpos + shift_pix, ypos - shift_pix ,ypos + shift_pix]
        # roi = np.ceil(roi)
        # roi = roi.astype(np.int32)

        (output_image,(xpos,ypos,width,height)) = self.__process_image(afield,template_image)
        # # 保存
        # defect_image = afield
        # output_image = template_image
        # output_image[roi[2]:roi[3],roi[0]:roi[1]] = defect_image[roi[2]:roi[3],roi[0]:roi[1]]
        label_name = target_filename + ".txt"
        with open(label_name,"w") as f:
            f.write(f"{defect_class} {xpos} {ypos} {width} {height}")
        
        # 保存超分辨（原图）
        cv2.imwrite(target_filename + "_origin.jpg",output_image)

        # 通过每个像素点代表的实际物理尺寸来计算缩放比比例
        scale_factor =source_density*1.0/target_density
        # 缩放电场/光强场到对应的大小
        scaled_field = cv2.resize(output_image, None, fx=scale_factor,# type: ignore
                                  fy=scale_factor, interpolation=cv2.INTER_LINEAR)  

        # 绘图
        logger.debug(f"printing max value of results:{np.max(total_results)}")
        cv2.imwrite(target_filename + ".jpg",scaled_field)
        logger.info("all target image saved completed!")
        
    def __process_image(self,defect_img,template_img,gap_length=10):
        diff_img = defect_img.astype(np.float32) - template_img.astype(np.float32)
        image_shape = template_img.shape
        # diff_img = (diff_img + 125)
        # diff_img = np.clip(diff_img, 0, 255).astype(np.uint8)

        gradX = cv2.Sobel(diff_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(diff_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        
        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)        
        defect_lowborder = np.max(gradient) * 0.35
        blurred = cv2.blur(gradient, (5, 5),borderType=cv2.BORDER_REFLECT) 
        (_, thresh) = cv2.threshold(blurred, defect_lowborder, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations=2,borderType=cv2.BORDER_ISOLATED)
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=5)

        # 找距离图像中心点最近的一个封闭区域
        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        min_dist = -1
        c = cnts[0]
        for conners in cnts:
            x,y,w,h = cv2.boundingRect(conners)
            rect_points = [(x, y),
                        (x + w, y),
                        (x + w, y + h),
                        (x, y + h)]
            distances = []
            for k in range(4):
                # 获取当前边的起点和终点
                p1 = rect_points[k]
                p2 = rect_points[(k + 1) % 4]

                # 计算点到当前边的距离
                distance = cv2.pointPolygonTest(np.array([p1, p2], np.int32),(image_shape[1]/2,image_shape[0]/2), True)
                distances.append(abs(distance))
            dist = min(distances)
            if dist < min_dist or min_dist == -1 :
                min_dist = dist
                c = conners

        # compute the rotated bounding box of the largest contour
        x,y,w,h=cv2.boundingRect(c)
        # img=cv2.rectangle(defect_img,(x,y),(x+w,y+h),(0,255,0),2)
        # 根据左上角坐标和长宽计算矩形的四个角点坐标
        rect_points = [(x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h)]

        # 开始扩展拼接缺陷图像
        outer_points = [(x-gap_length,y-gap_length),
                        (x+w+gap_length,y-gap_length),
                        (x+w+gap_length,y+h+gap_length),
                        (x-gap_length,y+h+gap_length)]

        output_img = template_img
        output_img[y:y+h,x:x+w] = defect_img[y:y+h,x:x+w]

        diff_img = diff_img
        x_lower_border = max(0,x - gap_length)
        x_upper_border = min(image_shape[1] - 1,x + w + gap_length - 1)
        y_lower_border = max(0,y - gap_length)
        y_upper_border = min(image_shape[0] - 1,y + h + gap_length - 1)
        for i in range(x_lower_border,x_upper_border):
            for j in range(y_lower_border,y_upper_border):
                if not (np.abs(i - x - w/2 + 0.5) < w/2 and np.abs(j - y - h/2 +0.5) < h/2):
                    # 计算点到矩形边界的距离
                    distances = []
                    distances2 = []
                    for k in range(4):
                        # 获取当前边的起点和终点
                        p1 = rect_points[k]
                        p2 = rect_points[(k + 1) % 4]
                        p11 = outer_points[k]
                        p22 = outer_points[(k+1)%4]

                        # 计算点到当前边的距离
                        distance = cv2.pointPolygonTest(np.array([p1, p2], np.int32),(i,j), True)
                        distances.append(abs(distance))
                        distance2 = cv2.pointPolygonTest(np.array([p11, p22], np.int32),(i,j), True)
                        distances2.append(abs(distance2))

                    # 获取最短距离
                    min_distance = min(distances)
                    min_distance2 = min(distances2)
                    # output_img[j,i] = 255
                    output_img[j,i] += diff_img[j,i]* (min_distance2)/(min_distance + min_distance2)
        xpos = (x + w/2)/image_shape[1]
        ypos = (y + h/2)/image_shape[0]
        width = w/image_shape[1]
        height = h/image_shape[1]
        return (output_img,(xpos,ypos,width,height))

    # def export_database_old(self, num_of_result, source_density, target_density,target_filename, vmax, is_light_intense=True, is_symmetry=False):
    #     # 开始提取
    #     # 先确定total_result的形状
    #     temp_result = self.resultbag.get_result(self.keys[0])
    #     field = (temp_result[num_of_result]['field'][0].conj() *
    #              temp_result[num_of_result]['field'][0]).sum(axis=2).real
    #     total_results = np.zeros(field.shape)
    #     logger.debug(f"total_result shape defined as {total_results.shape}")

    #     # 开始逐个提取结果
    #     for key in self.keys:
    #         result = self.resultbag.get_result(key)
    #         field = (result[num_of_result]['field'][0].conj() *
    #                  result[num_of_result]['field'][0]).sum(axis=2).real
    #         if is_light_intense:
    #             field = np.power(field, 2)
    #         total_results += field
    #         if is_symmetry and not (key['thetaphi'][0] == 0 and key['thetaphi'][1] == 0):
    #             field = np.rot90(field, 2)
    #             total_results += field
    #             logger.debug("key was rotated for symmetry")

    #     vmaxa = np.max(total_results) if vmax is None else vmax
    #     afield = (total_results/ vmaxa)*235
    #     afield = np.rot90(afield)

    #     # 通过每个像素点代表的实际物理尺寸来计算缩放比比例
    #     scale_factor =source_density*1.0/target_density
    #     # 缩放电场/光强场到对应的大小
    #     scaled_field = cv2.resize(afield, None, fx=scale_factor,# type: ignore
    #                               fy=scale_factor, interpolation=cv2.INTER_LINEAR)  

    #     # 绘图
    #     logger.debug(f"printing max value of results:{np.max(total_results)}")
    #     cv2.imwrite(target_filename,scaled_field)
    #     logger.info("all target image saved completed!")