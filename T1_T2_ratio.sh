for date_dir in [0-9]*/; do
    date_dir=${date_dir%/}
    echo "Processing T1/T2 for: $date_dir"
    
    cd "/data/users/hzhai/data/Neuroplastic/$date_dir" || continue
    
    for sub_dir in sub001 sub002; do
        [ ! -d "$sub_dir" ] && continue

	work_dir="${sub_dir}/T1_T2_ratio"
	if [ ! -f "${work_dir}/t1_iso1mm.nii"  ] || \
           [ ! -f "${work_dir}/t2_mx3d_sag_iso1mm.nii"  ]; then
            echo "WARNING: Missing files in ${date_dir}/${sub_dir}"
            continue
        fi
	t1="${work_dir}/t1_iso1mm.nii";
	t2="${work_dir}/t2_mx3d_sag_iso1mm.nii";

	# bias cor
	bias_cor_dir="${work_dir}/bias_cor"
	mkdir -p "$bias_cor_dir"

	N4BiasFieldCorrection -d 3 -i ${t1}                        -o ${bias_cor_dir}/T1_n4.nii.gz -s 8 -b [200] -c [50x50x50x50,0.000001]
	N4BiasFieldCorrection -d 3 -i ${bias_cor_dir}/T1_n4.nii.gz -o ${bias_cor_dir}/T1_n4.nii.gz -s 4 -b [200] -c [50x50x50x50,0.000001]
	N4BiasFieldCorrection -d 3 -i ${bias_cor_dir}/T1_n4.nii.gz -o ${bias_cor_dir}/T1_n4.nii.gz -s 2 -b [200] -c [50x50x50x50,0.000001]

	N4BiasFieldCorrection -d 3 -i ${t2}                        -o ${bias_cor_dir}/T2_n4.nii.gz -s 8 -b [200] -c [50x50x50x50,0.000001]
	N4BiasFieldCorrection -d 3 -i ${bias_cor_dir}/T2_n4.nii.gz -o ${bias_cor_dir}/T2_n4.nii.gz -s 4 -b [200] -c [50x50x50x50,0.000001]
	N4BiasFieldCorrection -d 3 -i ${bias_cor_dir}/T2_n4.nii.gz -o ${bias_cor_dir}/T2_n4.nii.gz -s 2 -b [200] -c [50x50x50x50,0.000001]
	
	echo "${date_dir}/${sub_dir}_biascor done"

	# register
	reg="${work_dir}/reg"
	mkdir -p "$reg"
	
	flirt -in ${bias_cor_dir}/T2_n4.nii.gz -ref ${bias_cor_dir}/T1_n4.nii.gz -out ${reg}/T2_2_T1.nii.gz -dof 6
	echo "${date_dir}/${sub_dir}_register done"
	
	# calc T1/T2_ratio
	calc="${work_dir}/calc"
	mkdir -p "$calc"

	bet ${reg}/T2_2_T1.nii.gz        ${calc}/T2 -m -f 0.25
	bet ${bias_cor_dir}/T1_n4.nii.gz ${calc}/T1 -m -f 0.25
	
        cp -r /data/users/hzhai/data/Neuroplastic/T1_T2_ratio.m $(pwd)
	
	matlab -nodesktop -nosplash -r "T1_T2_ratio('${calc}/T1.nii.gz','${calc}/T2.nii.gz','${calc}/T1_T2_ratio.nii.gz');quit;";
	echo "${date_dir}/${sub_dir}_calc done"

	# surface
	surface="${work_dir}/surface"
	fsdir="/data/users/hzhai/data/Neuroplastic/freesurfer_outputs";
	cmd="export SUBJECTS_DIR=${fsdir}";
	echo ${cmd}; eval ${cmd};
	subjnum="${date_dir}_${sub_dir}"
	infile="${bias_cor_dir}/T1_n4.nii.gz"
	
	bbregister --s ${subjnum} --6 --mov ${infile} --t1 \
	 --o ${surface}/T1_2_fsT1_6dofbbr.nii.gz --init-header \
	 --reg ${surface}/T1_avg_2_fsT1_6dofbbr.dat \

	for interp in trilinear; do
	for proj in 0.5 avg; do
	for hemi in lh rh; do

	mri_vol2surf --mov ${calc}/T1_T2_ratio.nii.gz \
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
	echo "${date_dir}/${sub_dir}_surface done"
    done
done
