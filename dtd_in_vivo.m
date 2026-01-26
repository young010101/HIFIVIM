%%
% Connect to data
ps = {};
ps.bp = '/data/users/cyang/dtd_subspace/';
ps.ip = fullfile(ps.bp, 'DATA', 'brain', 'cyang_20251212', 'NII'); % <- actual input data
ps.op = fullfile(ps.bp, 'processed', 'brain',  'cyang_20251212'); % <- store output here
ps.zp = fullfile(ps.bp, 'tmp'); % <- store temporary files here

if ((~exist(ps.bp, 'dir')) || ...
        (~exist(ps.ip, 'dir')))
    error('Data not found at specified folder. See instructions in header.')
end


msf_mkdir(ps.op); 
msf_mkdir(ps.zp); 

% Set options
opt = mdm_opt;
opt.do_overwrite = 1;
opt.verbose      = 1;

b_delta_lte = 1;
b_delta_ste = 0;

% Load linear and spherical data
f = @(nii_fn, b_delta) mdm_s_from_nii(fullfile(ps.ip, nii_fn), b_delta);

s = {...
    f('_LTE_2mmiso_PA_20251212195618_801.nii.gz', b_delta_ste), ...
    f('_STE_2mmiso_PA_20251212195618_701.nii.gz', b_delta_lte)};

% Merge into one file
s = mdm_s_merge(s, ps.op, 'FWF_phan', opt);

% Also make a powder-averaged data set for visualization
% s.xps = rmfield(s.xps, 's_ind');
% s_pa = mdm_s_powder_average(s, ps.op, opt);

%%
% Smooth the data a bit
opt.filter_sigma = 0.6;

% Do covariance analysis using all of the data
tic;
% dtd_gamma_pipe(s, ps.op, opt);
dtd_codivide_pipe(s, ps.op, opt);
toc;