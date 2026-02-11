# calc="/data/users/cyang/dtd_subspace/0119_ngc/processed/brain_2mmiso/lte_sense_rot180_norm_removeb0_stes_basis4_real_removeb0"
calc="/data/users/cyang/dtd_subspace/0119_ngc/processed/brain/dicom"

work_dir="/data/users/cyang/repos/HIFIVIM"
surface="${work_dir}/surface"
fsdir="/data/users/cyang/freesurfer_test/recon";
cmd="export SUBJECTS_DIR=${fsdir}";
echo ${cmd}; eval ${cmd};
subjnum="cyang"
infile="/data/users/cyang/DICOM_YANGCHENG/NII/_b0_2mmiso_AP_20260118122029_301.nii.gz"

# bbregister --s ${subjnum} --6 --mov ${infile} --t1 \
#     --o ${surface}/T1_2_fsT1_6dofbbr.nii.gz --init-header \
#     --reg ${surface}/T1_avg_2_fsT1_6dofbbr.dat \
for interp in trilinear; do
for proj in 0.5 avg; do
for hemi in lh rh; do

mri_vol2surf --mov ${calc}/dtd_gamma_MD.nii.gz \
--surf white --projfrac -${proj} \
--hemi ${hemi} \
--reg ${surface}/T1_avg_2_fsT1_6dofbbr.dat \
--cortex \
--interp ${interp} \
--o ${surface}/${hemi}_proj_T1_T2_ratio_p${proj}_${interp}.mgh 

mri_surf2surf --srcsubject ${subjnum} --trgsubject fsaverage --hemi ${hemi} \
--sval ${surface}/${hemi}_proj_T1_T2_ratio_p${proj}_${interp}.mgh \
--tval ${surface}/${hemi}_proj_T1_T2_ratio_p${proj}_${interp}_2avg.mgh 

done
done
done
