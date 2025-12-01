# %%
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# Phantom 数据值范围：如果是T1,应该在 0.02s 到 1s之间
'''
fat 300
wm 800
gm 1400
CSF 4000
water 4000
'''
# %% =======================t1 crop============================
phan_t1_img=nib.load('../Phantom/Phantom_T1.nii.gz')
phan_t1_data = phan_t1_img.get_fdata()

# %%
print(np.max(phan_t1_data))
print(phan_t1_data.min())

plt.imshow(phan_t1_data[:,:,90])
# %%
plt.hist(phan_t1_data[:,:,0])
# %%=====================t1 full==============================
path_t1_icbm='../local/t1_icbm_normal_1mm_pn3_rf20.mnc.gz'
t1_icbm_normal_img = nib.load(path_t1_icbm)
t1_icbm_normal=t1_icbm_normal_img.get_fdata()
# %%
# nib.save(t1_icbm_normal_img,'../local/t1_icbm_normal_1mm_pn3_rf20.nii.gz')

def brainweb_to_ras(data):
    # data shape: (181, 217, 181)  # (Z:脚→头, Y:后→前, X:左→右)
    data = np.rot90(data, k=1, axes=(0, 2))   # 先绕Y轴转90° → 冠状
    data = np.flip(data, axis=0)              # 上下翻转
    data = np.flip(data, axis=2)              # 左右翻转（可选，根据需要）
    return data

data_ras = brainweb_to_ras(t1_icbm_normal)

path_t1_icbm_data_ras='../local/t1_icbm_ras.nii.gz'
nib.save(nib.Nifti1Image(data_ras, affine=t1_icbm_normal_img.affine),path_t1_icbm_data_ras)
#%%
plt.imshow(abs(t1_icbm_normal[90,:,:]))
# %%===================phase==========================
phase_esti_img=nib.load('../Phantom/phase_estimates.nii.gz')
phase_esti = phase_esti_img.get_fdata()
# %%
print(phase_esti.shape)
print(phase_esti.max())
print(phase_esti.min())
plt.imshow(abs(phase_esti[:,:,0]))
plt.figure()
plt.imshow(np.angle(phase_esti[:,:,0]))
plt.colorbar()
fig, axes = plt.subplots(ncols=5, nrows=3)
for i, ax in enumerate(axes.flat):
    ax.imshow(abs(phase_esti[:,:,i]))
    ax.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace=0,hspace=0)

# %%=====================sens===========================
sens_img=nib.load('../Phantom/sens_maps.nii.gz')
sens_data =sens_img.get_fdata()
print(sens_data.shape)
print(sens_data.max())
print(sens_data.min())
plt.imshow(abs(sens_data[:,:,0,0]))
plt.imshow(np.angle(sens_data[:,:,0,0]))
plt.colorbar()

# %% =========== csf 矩形 =======================
phantom=abs(phase_esti[:,:,0])
sq_mask=np.zeros_like(phantom)
sq_mask[62:103,65:101]=1
plt.imshow(phantom * sq_mask)
plt.figure()
plt.imshow(phantom * (1-sq_mask))

# %%=======================regi==============================
import SimpleITK as sitk
output_2d_path = 'T2_slice85_to_T1_slice85.nii.gz'
slice_index=90
fixed_3d = sitk.ReadImage('../Phantom/phase_estimates.nii.gz', sitk.sitkFloat32)

path_t1_icbm_nii='../local/t1_icbm_normal_1mm_pn3_rf20.nii.gz'
moving_3d = sitk.ReadImage(path_t1_icbm_data_ras, sitk.sitkFloat32)
fixed_2d = fixed_3d[:,:,0]
moving_2d = moving_3d[:,:,slice_index]
fixed=fixed_2d
moving=moving_2d
print(fixed.GetDimension())
print(fixed.GetSize())
print(fixed.GetSpacing())
print(fixed.GetDirection())
print(moving.GetDimension())
print(moving.GetSize())
print(moving.GetSpacing())
print(moving.GetDirection())
# Step1: 刚性+仿射（快速对齐）
transform = sitk.CenteredTransformInitializer(fixed, moving, 
                                                        sitk.Euler2DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)
# %%
reg = sitk.ImageRegistrationMethod()
reg.SetMetricAsMattesMutualInformation(50)
reg.SetOptimizerAsGradientDescent(learningRate=2.0, numberOfIterations=200)
reg.SetOptimizerScalesFromPhysicalShift()
reg.SetInitialTransform(transform, inPlace=False)
reg.SetInterpolator(sitk.sitkLinear)

final_transform = reg.Execute(fixed_2d, moving_2d)

# 5. 重采样
warped_2d = sitk.Resample(moving_2d, fixed_2d, final_transform,
                          sitk.sitkLinear, 0.0, moving_2d.GetPixelID())

# 6. 保存 + 可视化
sitk.WriteImage(warped_2d, output_2d_path)

# 转 numpy 显示
f = sitk.GetArrayViewFromImage(fixed_2d)
m = sitk.GetArrayViewFromImage(moving_2d)
w = sitk.GetArrayViewFromImage(warped_2d)

plt.figure(figsize=(15,5))
plt.subplot(1,4,1); plt.imshow(f, cmap='gray'); plt.title(f'Fixed slice {slice_index}'); plt.axis('off')
plt.subplot(1,4,2); plt.imshow(m, cmap='gray'); plt.title('Moving original'); plt.axis('off')
plt.subplot(1,4,3); plt.imshow(w, cmap='gray'); plt.title('After 2D registration'); plt.axis('off')
plt.subplot(1,4,4); plt.imshow(0.6*f + 0.4*w, cmap='gray'); plt.title('Overlay'); plt.axis('off')
plt.tight_layout()
plt.show()

print(f"第 {slice_index} 层 2D 配准完成！结果已保存")
# %%
