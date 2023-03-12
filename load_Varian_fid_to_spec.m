clear;
[RE1,IM1] =WeiF_LoadFid('exp/Pure_shift_archive_Varian/data/25_UoM-Workshop_Quinine_50mM_rtZS_1-proc_2017-09-06_v500_UoM_1d_rt_PS_dmso_25C_HCX_01.fid/fid',1);

FID=RE1+1i*IM1;
FID=FID';
spec=fftshift(fft(FID,4096));
theta=auto_phase(spec,2);
spec=spec.*exp(1i*theta);
spec=real(spec);
spec=spec/max(spec);
spec=fliplr(spec);

figure();
plot(spec);

exp_txt = fopen('exp/exp.txt','wt');
[m,n]=size(spec);
for i=1:1:m
    for j=1:1:n
        if j==n
            fprintf(exp_txt,'%.16f\n',spec(i,j));
        else
            fprintf(exp_txt,'%.16f\t',spec(i,j));
        end
    end
end 
