
import numpy as np
import k3d


def volume3D(field, level, size='nan'):
    """
    
    
    """
    if size == 'nan':
        size = np.array([600, 600, 600])
        
    volume = k3d.volume(field.astype(np.float32), 
                        alpha_coef=level,
                        color_range=[-0.4, 0.4], 
                        color_map=(np.array(k3d.colormaps.paraview_color_maps.Cool_to_Warm_Extended).reshape(-1,4) 
                                * np.array([1.0,1.0,1.0,1.0])).astype(np.float32)
                    )

    volume.opacity_function  = [0.        , 0.        , 0.21327923, 0.98025   , 0.32439035,
           0.        , 0.5       , 0.        , 0.67560965, 0.        ,
           0.74537706, 0.9915    , 1.        , 0.        ]


    volume.transform.bounds = [-size[0]/2,size[0]/2,
                               -size[1]/2,size[1]/2,
                               -size[2]/2,size[2]/2]

    return volume 

def filed_evolution(fields_t, snap_0, ngrid=(50, 50, 50)):
    rho0 = fields_t['snap_000']['pot0'][:].reshape(ngrid)

    for t in range(0, 300, 7):
        rhoall = fields1['snap_{:03d}'.format(t)]['pot'][:].reshape((50, 50, 50)).T
        img = (((rho0+rhoall) / (rho0)) - 1 ) # * 7
        
        psi_t[str(float(t))] = img.astype(np.float32)
    return psi_t

def make_volume_render(densities, volumes, size, display=False, **kwargs):
    # Contours
    vols = []
    plot3d = k3d.plot(height=800)
    

    for volume in volumes:
        vol = volume3D(field, level=200.0, size=size)
        vol.color_range = [-0.2, 0.2]
        vols.append(vol)
        plot3d += vol
    
    # Labels
    if label in kwargs:
        plt_tp_text = k3d.text2d(kwargs['label'],
                            position=[0.01, 0.05],
                            reference_point='lc',
                            is_html=True,
                            color=0x3f71d8)
        plot3d += plt_tp_text

    # Orbit 
    if orbit in kwargs:
        plt_streamlines = k3d.line(np.array([sat_orbit[:40,1], sat_orbit[:40,0], sat_orbit[:40,2]]).T, 
                                   color=222,
                                   width=0.00007)
    
    if display == True:
        plot3d.display()
    
    if 

