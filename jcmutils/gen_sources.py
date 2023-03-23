import numpy as np
from .logger import logger


def gen_kohler_sources(maxtheta, phi0, spacing, lambda0, flag_is_symmetry=False):
    logger.info("generating kohler sources")
    logger.debug(f"received parameters was :maxtheta-{maxtheta}-phi0-{phi0}-spacing-{spacing}-lambda0-{lambda0}-symmetry-{flag_is_symmetry}")

    maxtheta = np.deg2rad(maxtheta)
    phi0 = np.deg2rad(phi0)

    # 按spacing的间隔生成候选
    candidate = np.linspace(-1, 1, spacing)
    f=1/np.tan(maxtheta)

    # 提取在圆形孔径光阑内的平面上的等距点
    coordinate = []
    for mu in candidate:
        for nu in candidate:
            if mu**2 + nu**2 <= 1:
                if flag_is_symmetry:
                    if (mu < 0 and nu <=0) or (mu >=0 and nu < 0):
                        continue
                coordinate.append([mu, nu])
    logger.debug(f"got coordinates done, {len(coordinate)} coordinates are generated")

    # 计算刚才得到的孔径光阑上的点对应的入射平面波
    keys = []
    for mu,nu in coordinate:
        phi = 0
        theta = np.abs(np.arctan(np.sqrt(mu**2+nu**2)/f))
        if mu > 0 and nu >= 0:
            phi = np.arctan(nu/mu)
        if mu <=0 and nu > 0:
            phi = np.pi/2 if mu == 0 else np.arctan(nu/mu) + np.pi
        if mu < 0 and nu <=0:
            phi = np.arctan(nu/mu) + np.pi
        if mu >=0 and nu < 0:
            phi = np.pi*3/2 if mu == 0 else np.arctan(nu/mu) + np.pi*2
        keys.append({'thetaphi': [theta, phi], 'lambda0': lambda0})
    # 通过透镜后的线偏振方向在入射平面中的投影直接就是p光，垂直于入射平面的就是s光
    # p方向的分量就是cos(PHI-psi0),s方向的分量就是sin(PHI-psi0),PHI是指入射光方位角。psi0是偏振方向与x轴夹角
    for key in keys:
        key['SP'] = [np.sin(key['thetaphi'][1] - phi0),np.cos(key['thetaphi'][1] - phi0)]
        key['thetaphi'][0] = np.rad2deg(key['thetaphi'][0])
        key['thetaphi'][1] = np.rad2deg(key['thetaphi'][1])
        logger.debug(f"key->{key} was successfully generated")
    
    logger.debug("kohler sources generate done")

    return keys
