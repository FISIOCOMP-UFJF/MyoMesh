import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
from nibabel import load # type: ignore
from cv2 import dilate # type: ignore
from skimage.measure import find_contours # type: ignore

def resample_closed_curve(points, num_points=80):
    points = np.asarray(points)
    
    # Close the curve if needed
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    # Compute cumulative distances (arc lengths)
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    # Total length of the curve
    total_length = arc_lengths[-1]
    
    # Create uniform arc lengths
    target_lengths = np.linspace(0, total_length, num_points, endpoint=False)
    
    # Interpolate in x and y separately
    interp_x = interp1d(arc_lengths, points[:, 0], kind='cubic')
    interp_y = interp1d(arc_lengths, points[:, 1], kind='cubic')
    
    resampled_x = interp_x(target_lengths)
    resampled_y = interp_y(target_lengths)
    
    return np.stack([resampled_x, resampled_y], axis=1)

data_dir = os.getcwd()
data_list = sorted(os.listdir(f'{data_dir}/input'))
section = ['ED', 'ES']

for data in data_list:
    os.makedirs(os.path.join(f'{data_dir}/output/{data}', 'contours-txt'), exist_ok=True)
    os.makedirs(os.path.join(f'{data_dir}/output/{data}', 'contours-png'), exist_ok=True)
    try:
        mat_path = f'{data_dir}/input/{data}/{data}.mat'
        mat = loadmat(mat_path)
        print(f'  Reading {mat_path} ...')
    except FileNotFoundError:
        print('  Erro: Arquivo .MAT não encontrado.')

    for fr in section:
        # Lendo a imagem
        image_name = '{0}/{1}.nii.gz'.format(f'{data_dir}/input/{data}', f'seg_sa_{fr}')
        if not os.path.exists(image_name):
            print('   Directory {0} does not contain an image with file '
                    'name {1}. Skip.'.format(data_dir, os.path.basename(image_name)))
            continue
        print('  Reading {} ...'.format(image_name))

        nim = load(image_name)
        image = nim.get_fdata()
        X, Y, Z = image.shape
        endox = np.full((80,1,Z), np.nan)
        endoy = np.full((80,1,Z), np.nan)
        rvendox = np.full((80,1,Z), np.nan)
        rvendoy = np.full((80,1,Z), np.nan)
        rvepix = np.full((80,1,Z), np.nan)
        rvepiy = np.full((80,1,Z), np.nan)
        
        for frame in range(Z):
            # Convertendo a imagem para 2D
            slice_2d = image[:, :, frame]
            # Definindo tamanho da plotagem
            fig, ax = plt.subplots(figsize=(Y / 25, X / 25))

            # Endo
            mask = (slice_2d == 1)
            # Encontrando os contornos na máscara
            contours = find_contours(mask, level=0.5)
            # Salvando os contornos em um array numpy
            if contours:
                for contour in contours:
                    if len(contour) > 2:
                        new_contours = resample_closed_curve(contour, 80)
                        for ind, point in enumerate(new_contours):
                            endox[ind,0,frame] = point[1]
                            endoy[ind,0,frame] = point[0]
            # Plotando os contornos com os pontos originais
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='red')
                
            # RVEndo
            mask = (slice_2d == 3)
            # Encontrando os contornos na máscara
            contours = find_contours(mask, level=0.5)
            # Salvando os contornos em um array numpy
            if contours:
                for contour in contours:
                    if len(contour) > 2:
                        new_contours = resample_closed_curve(contour, 80)
                        for ind, point in enumerate(new_contours):
                            rvendox[ind,0,frame] = point[1]
                            rvendoy[ind,0,frame] = point[0]
            # Plotando os contornos com os pontos originais
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='red')

            # RVEpi
            kernel_size = 3
            # Cria uma matriz quadrada de tamanho kernel_size x kernel_size
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # Técnica de dilatação
            mask = dilate(mask.astype(np.uint8), kernel, iterations=1)
            # Subtraindo interseção entre a máscara 2
            mask = mask & ~(slice_2d == 2)
            # Adicionando padding na imagem original
            slice_2d += mask
            # Extraindo contornos da segmentação completa com adição do padding
            contours = find_contours(slice_2d, level=0.1)
            # Salvando os contornos em um array numpy
            if contours:
                for contour in contours:
                    if len(contour) > 2:
                        new_contours = resample_closed_curve(contour, 80)
                        for ind, point in enumerate(new_contours):
                            rvepix[ind,0,frame] = point[1]
                            rvepiy[ind,0,frame] = point[0]
            # Plotando os contornos com os pontos originais
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='blue')

            # Salvando os contornos em imagens .jpg
            contour_image_path = os.path.join(f'{data_dir}/output/{data}/contours-png', f'{fr}_{frame}.png')
            plt.axis('off')
            plt.savefig(contour_image_path, pad_inches=0)
            plt.close(fig)

            # Salvando os contornos em arquivos .txt 
            txt_path = os.path.join(f'{data_dir}/output/{data}/contours-txt', f'{fr}_{frame}_Endo.txt')
            with open(txt_path, 'w') as txt_file:
                for ind in range(80):
                    if np.isnan(endox[ind,0,frame]):
                        txt_file.write('0 0 0\n')
                    else:
                        txt_file.write(f'{endox[ind,0,frame]:.6f} {endoy[ind,0,frame]:.6f} 0\n')

            txt_path = os.path.join(f'{data_dir}/output/{data}/contours-txt', f'{fr}_{frame}_RVEndo.txt')
            with open(txt_path, 'w') as txt_file:
                for ind in range(80):
                    if np.isnan(rvendox[ind,0,frame]):
                        txt_file.write('0 0 0\n')
                    else:
                        txt_file.write(f'{rvendox[ind,0,frame]:.6f} {rvendoy[ind,0,frame]:.6f} 0\n')

            txt_path = os.path.join(f'{data_dir}/output/{data}/contours-txt', f'{fr}_{frame}_RVEpi.txt')
            with open(txt_path, 'w') as txt_file:
                for ind in range(80):
                    if np.isnan(rvepix[ind,0,frame]):
                        txt_file.write('0 0 0\n')
                    else:
                        txt_file.write(f'{rvepix[ind,0,frame]:.6f} {rvepiy[ind,0,frame]:.6f} 0\n')

        # Reescrevendo arquivo .MAT
        if 'setstruct' in mat: 
            mat['setstruct']['EndoX'][0][0] = endox
            mat['setstruct']['EndoY'][0][0] = endoy
            mat['setstruct']['RVEndoX'][0][0] = rvendox
            mat['setstruct']['RVEndoY'][0][0] = rvendoy
            mat['setstruct']['RVEpiX'][0][0] = rvepix
            mat['setstruct']['RVEpiY'][0][0] = rvepiy
            savemat(f'{data_dir}/output/{data}/{data}_editado.mat', mat)
            print('  Salvando arquivo .MAT ...')
        else:
            print('  Erro: Variável "setstruct" não encontrada no arquivo .MAT')
            
print ('  Find Contours done.')
