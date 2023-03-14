import numpy as np


def gen_kohler_sources(maxtheta, phi0, spacing, lambda0, flag_is_symmetry=False):
    maxtheta = np.deg2rad(maxtheta)
    phi0 = np.deg2rad(maxtheta)
    # 按spacing的间隔生成候选
    candidate = np.linspace(-np.sin(maxtheta), np.sin(maxtheta), spacing)

    # 提取在圆形孔径光阑内的平面上的等距点
    coordinate = []
    for sx in candidate:
        for sy in candidate:
            if sx**2 + sy**2 <= np.sin(maxtheta)**2:
                coordinate.append([sx, sy])

    # 计算刚才得到的孔径光阑上的点对应的入射平面波
    keys = []
    for i in coordinate:
        theta = 0
        phi = 0
        if i[0] > 0 and i[1] >= 0:
            phi = np.arctan(i[1]/i[0])
            theta = np.arcsin(i[0]/np.cos(phi))
        if i[0] <= 0 and i[1] > 0:
            phi = np.pi/2 if i[0] == 0 else np.arctan(i[1]/i[0]) + np.pi
            theta = np.arcsin(i[1]/np.sin(phi))
        if (flag_is_symmetry is False) and (i[0] < 0 and i[1] <= 0):
            phi = np.arctan(i[1]/i[0]) + np.pi
            theta = np.arcsin(i[0]/np.cos(phi))
        if (flag_is_symmetry is False) and (i[0] >= 0 and i[1] < 0):
            phi = np.pi*3/2 if i[0] == 0 else np.arctan(i[1]/i[0]) + 2*np.pi
            theta = np.arcsin(i[1]/np.sin(phi))
        keys.append({'thetaphi': [theta, phi], 'lambda0': lambda0})
    # 通过透镜后的线偏振方向在入射平面中的投影直接就是p光，垂直于入射平面的就是s光
    # p方向的分量就是cos(PHI-psi0),s方向的分量就是sin(PHI-psi0),PHI是指入射光方位角。psi0是偏振方向与x轴夹角
    for key in keys:
        key['SP'] = [np.sin(key['thetaphi'][1] - phi0),np.cos(key['thetaphi'][1] - phi0)]
        key['thetaphi'][0] = np.rad2deg(key['thetaphi'][0])
        key['thetaphi'][1] = np.rad2deg(key['thetaphi'][1])

    return keys
