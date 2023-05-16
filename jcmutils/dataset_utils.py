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
        template_image = cv2.imread(template_path)
        origin_image_size = template_image.shape
        
        # 确定缺陷类别
        defect_class = 0
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

        # 确定缺陷在原始图像中的位置
        xpos = self.keys[0]['defectpos'][0] * 1.0/( origin_size['x'][1] - origin_size['x'][0]) * origin_image_size[0]
        ypos = origin_image_size-(self.keys[0]['defectpos'][1] * 1.0/(origin_size['y'][1] - origin_size['y'][0]) * origin_image_size[1])
        shift_pix = defect_size*1.0/source_density
        roi = [xpos - shift_pix , xpos + shift_pix, ypos - shift_pix ,ypos + shift_pix]
        roi = np.ceil(roi)

        # 保存
        defect_image = afield
        output_image = template_image
        output_image[roi[0]:roi[1],roi[2]:roi[3]] = defect_image[roi[0]:roi[1],roi[2]:roi[3]]
        label_name = target_filename + ".txt"
        with open(label_name,"w") as f:
            f.write(f"{defect_class} {xpos/origin_image_size[0]} {ypos/origin_image_size[1]} {defect_size*2/origin_image_size[0]} {defect_size*2/origin_image_size[1]}")
        
        # 保存超分辨（原图）
        cv2.imwrite(target_filename + "_origin.jpg",output_image)

        # 通过每个像素点代表的实际物理尺寸来计算缩放比比例
        scale_factor =source_density*1.0/target_density
        # 缩放电场/光强场到对应的大小
        scaled_field = cv2.resize(afield, None, fx=scale_factor,# type: ignore
                                  fy=scale_factor, interpolation=cv2.INTER_LINEAR)  

        # 绘图
        logger.debug(f"printing max value of results:{np.max(total_results)}")
        cv2.imwrite(target_filename + ".jpg",scaled_field)
        logger.info("all target image saved completed!")
        

    def export_database_old(self, num_of_result, source_density, target_density,target_filename, vmax, is_light_intense=True, is_symmetry=False):
        # 开始提取
        # 先确定total_result的形状
        temp_result = self.resultbag.get_result(self.keys[0])
        field = (temp_result[num_of_result]['field'][0].conj() *
                 temp_result[num_of_result]['field'][0]).sum(axis=2).real
        total_results = np.zeros(field.shape)
        logger.debug(f"total_result shape defined as {total_results.shape}")

        # 开始逐个提取结果
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

        vmaxa = np.max(total_results) if vmax is None else vmax
        afield = (total_results/ vmaxa)*235
        afield = np.rot90(afield)

        # 通过每个像素点代表的实际物理尺寸来计算缩放比比例
        scale_factor =source_density*1.0/target_density
        # 缩放电场/光强场到对应的大小
        scaled_field = cv2.resize(afield, None, fx=scale_factor,# type: ignore
                                  fy=scale_factor, interpolation=cv2.INTER_LINEAR)  

        # 绘图
        logger.debug(f"printing max value of results:{np.max(total_results)}")
        cv2.imwrite(target_filename,scaled_field)
        logger.info("all target image saved completed!")