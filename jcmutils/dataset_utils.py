from .logger import logger
import numpy as np
import os
import jcmwave
import cv2


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

    def export_database(self, num_of_result, source_density, target_density,target_filename, vmax, is_light_intense=True, is_symmetry=False):
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
