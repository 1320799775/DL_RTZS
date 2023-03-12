close all;
clear all;
clc;

num_data=40000;
label_txt = fopen('dataset/label.txt','wt');
data_txt = fopen('dataset/rtzs.txt','wt');
range_txt = fopen('dataset/range.txt','wt');

for num_data0=1:num_data

    r_sw=2^(unidrnd(2)-1);          % r_sw is 1,2
    sw=2048*r_sw;                   % sw is 2048,4096
    wnum=round(20 + rand(1)*20);    % peak number is 20 ~ 40
    w=[];
    for num0=1:wnum
        find_s=1;
        for i=1:100                 % different chemical shift
            add_s=round((480*rand(1,1)+15)*r_sw)*4;   % chemical shift
            find_s=ismember(add_s,w);
            if find_s==0
                break;
            end
        end
        w=[w,add_s];
    end
    jstrong=3;                         
    Jnum = randi(jstrong, 1, wnum);                       
    J = 12*rand(wnum, jstrong)+2;                         % J-coupling intensity
    pow = randi(4, wnum, jstrong) - ones(wnum, jstrong);  % coupling spin number
 
    T2=[];
    rate_T2=0.9;
    for ww=1:wnum 
        if rand(1)<0.9
            T2=[T2,0.47*rand(1,1)+0.13];
        else
            T2=[T2,0.4*rand(1,1)+0.6];
        end
    end
    T2_0=1;                         % T2: label of double peaks
    T2_1=0.25+0.1*rand(1,1);        % T2: double peaks
    
    na=round(7*rand(1,1)+3)*10;     % chunk time
    nd=round(3*rand(1,1)+3)*10;     % chunk number
    dd=round(35*rand(1,1)+10)*10;   % unsampling time

    snr=8.5*0.0001*rand(1,1);       % noise intensity coefficient
    np=4096;
    np0=(na+dd)*nd*2;
    deltat = 1/sw;
    t = [0:deltat:deltat*(np0-1)];

    %%%%%%%%%%%%%%%%%%%%strength of peak
    str0 = 0.23;
    str1 = 0.39;
    str2 = 0.53;
    str3 = 0.78;
    strong = [round(3*rand(1,1)+17)];
    wstrong = 20;
    for ww=1:(wnum-1)
        str=rand(1);
        if str>str3
            strong1 = round(3*rand(1,1)+17);
        elseif str>str2
            strong1 = round(3*rand(1,1)+10);
        elseif str>str1 
            strong1 = round(2*rand(1,1)+5);
        elseif str>str0
            strong1 = round(2*rand(1,1)+2);  
        else
            strong1 = 1;  
        end
        strong=[strong,strong1];
    end

    %%%%%%overlap weak peak
    signal = zeros(1,length(t));
    signal1=zeros(1,length(t));
    find_s=1;
    for i=1:100  
        shift_strong=round((480*rand(1,1)+15)*r_sw)*4;
        find_s=ismember(shift_strong,w);
        if find_s==0
            break;
        end
    end
    intense_strong =  round(2*rand(1,1)+2);
    shift_strong1 = round(3*rand(1,1)+4);
    tt_strong1 = intense_strong*exp(1i*(2*pi*(shift_strong)+2*pi)*t).*exp(-t/T2_0);
    tt_strong2 = intense_strong*exp(1i*(2*pi*(shift_strong+shift_strong1)+2*pi)*t).*exp(-t/T2_0);
    tt_strong1_0 = 6*exp(1i*(2*pi*(shift_strong)+2*pi)*t).*exp(-t/T2_0);
    tt_strong2_0 = 6*exp(1i*(2*pi*(shift_strong+shift_strong1)+2*pi)*t).*exp(-t/T2_0);

    %%%%%%overlap strong peak
    find_s=1;
    for i=1:100 
        shift_strong2=round((480*rand(1,1)+15)*r_sw)*4;
        find_s=ismember(shift_strong2,[w,shift_strong,shift_strong+4,shift_strong-4,shift_strong+8,shift_strong-8]);
        if find_s==0
            break;
        end
    end
    intense_strong1 =  round(3*rand(1,1)+14);
    intense_strong2 =  round(3*rand(1,1)+14);
    shift_strong3 = round(2*rand(1,1)+4);
    tt_strong3 = intense_strong1*exp(1i*(2*pi*(shift_strong2)+2*pi)*t).*exp(-t/T2_0);
    tt_strong4 = intense_strong2*exp(1i*(2*pi*(shift_strong2+shift_strong3)+2*pi)*t).*exp(-t/T2_0);
    tt_strong3_0 = 18*exp(1i*(2*pi*(shift_strong2)+2*pi)*t).*exp(-t/T2_0);
    tt_strong4_0 = 18*exp(1i*(2*pi*(shift_strong2+shift_strong3)+2*pi)*t).*exp(-t/T2_0);

    %%%%%%%probability of high double peak
    proba=rand();
    proba1=1-0.3;
    if proba>proba1
        shift_end=[w,shift_strong,(shift_strong1+shift_strong),shift_strong2,(shift_strong2+shift_strong3)];
    else
        shift_end=[w,shift_strong,(shift_strong1+shift_strong)];
        tt_strong3=zeros(1,length(t));
        tt_strong4=zeros(1,length(t));
        tt_strong3_0=zeros(1,length(t));
        tt_strong4_0=zeros(1,length(t));
    end
    
    %%%%%%%%%%%%label
    for ww=1:wnum
        intense=strong(ww);
        tt = intense*exp(1i*(2*pi*(w(ww))+2*pi)*t).*exp(-t/T2(ww));
        signal = signal + tt;
    end
    signal = signal+tt_strong1+tt_strong2+tt_strong3+tt_strong4;
    fid_1d = signal;
    spec_1d = real(fft(fid_1d,np));
    spec_1d=spec_1d/max(abs(spec_1d));
    spec_Y=real(spec_1d);

    %%%%%%%%%%%%%% rtZS
    realTime_fid = zeros(1,na*nd+na/2);
    J_index = 1;
    for ww=1:wnum
        t1 = 0:na/2-1;
        t1=t1/sw;
        t2 = -na/2:na/2-1;
        t2=t2/sw;
        J_terms = 1;
        jnum = Jnum(ww);
        for z=1:jnum
            J_terms = J_terms.*cos(pi*J(ww, z)*t1).^pow(ww, z);   
        end  
        intense = strong(ww);
        realTime_fid_single = intense*J_terms.*exp(1i*(2*pi*(w(ww)))*t1).*exp(-t1/T2(ww));
        for index=1:nd
            t3 = na*(1/2+index-1):na*(1/2+index)-1;
            t3=t3/sw;
            t4 = na*(1/2+index-1)+dd*index:na*(1/2+index)+dd*index-1;
            t4=t4/sw;

            J_terms = 1;
            for z=1:jnum
                J_terms = J_terms.*cos(pi*J(ww, z)*t2).^pow(ww, z);   
            end  
            chunk = intense.*J_terms.*exp(1i*2*pi*(w(ww))*t3).*exp(-t4/T2(ww));
            realTime_fid_single = [realTime_fid_single,chunk];
        end
        realTime_fid = realTime_fid+realTime_fid_single;
    end

    %%%%%%%%%%%%%%%%%%%%%%overlap peaks
    t1 = 0:na/2-1;
    t1=t1/sw;
    t2 = -na/2:na/2-1;
    t2=t2/sw;
    tt_strong1_1 = intense_strong*exp(1i*(2*pi*(shift_strong)+2*pi)*t1).*exp(-t1/T2_1);
    tt_strong2_1 = intense_strong*exp(1i*(2*pi*(shift_strong+shift_strong1)+2*pi)*t1).*exp(-t1/T2_1);
    %%%%%%strong double peak
    if proba>proba1
        tt_strong3_1 = intense_strong1*exp(1i*(2*pi*(shift_strong2)+2*pi)*t1).*exp(-t1/T2_0);
        tt_strong4_1 = intense_strong2*exp(1i*(2*pi*(shift_strong2+shift_strong3)+2*pi)*t1).*exp(-t1/T2_0);
    else
        tt_strong3_1=zeros(1,length(t1));
        tt_strong4_1=zeros(1,length(t1));
    end
    for index=1:nd
        t3 = na*(1/2+index-1):na*(1/2+index)-1;
        t3=t3/sw;
        t4 = na*(1/2+index-1)+dd*index:na*(1/2+index)+dd*index-1;
        t4=t4/sw;
        chunk1_1 = intense_strong*exp(1i*(2*pi*(shift_strong)+2*pi)*t3).*exp(-t4/T2_1);
        chunk2_1 = intense_strong*exp(1i*(2*pi*(shift_strong+shift_strong1)+2*pi)*t3).*exp(-t4/T2_1);
        if proba>proba1
            chunk3_1 = intense_strong1*exp(1i*(2*pi*(shift_strong2)+2*pi)*t3).*exp(-t4/T2_0);
            chunk4_1 = intense_strong2*exp(1i*(2*pi*(shift_strong2+shift_strong3)+2*pi)*t3).*exp(-t4/T2_0);
        else
            chunk3_1=zeros(1,length(t3));
            chunk4_1=zeros(1,length(t3));
        end
        tt_strong1_1 = [tt_strong1_1,chunk1_1];
        tt_strong2_1 = [tt_strong2_1,chunk2_1];
        tt_strong3_1 = [tt_strong3_1,chunk3_1];
        tt_strong4_1 = [tt_strong4_1,chunk4_1];
    end
    realTime_fid = (realTime_fid+tt_strong1_1+tt_strong2_1+tt_strong3_1+tt_strong4_1);

    spec_realtime0=fft(realTime_fid,np);
    spec_realtime_clear=spec_realtime0/max(real(spec_realtime0));
    realTime_fid=ifft(spec_realtime_clear);
    realTime_fid0=realTime_fid+snr*(randn(1,length(realTime_fid))+1i*randn(1,length(realTime_fid)));
    spec_realtime = fft(realTime_fid0,np);

    spec_realtime=spec_realtime/max(abs(spec_realtime));
    spec_realtime=real(spec_realtime);
    spec_realtime=spec_realtime/max(abs(spec_realtime));
    spec_X=real(spec_realtime);

    %%%%%%%%%%peak range matrix
    spec_H2=zeros(1,np);
    spec_Hi=ones(1,length(shift_end));
    shift_end=shift_end.*2./r_sw;
    for shift_wide=1:32     % 32 points are enough
        spec_H2(sub2ind(size(spec_H2),spec_Hi,(shift_end-16+shift_wide)))=1;
    end
    spec_H=spec_H2;

%     figure;subplot(311);plot(spec_X);subplot(312);plot(spec_Y);subplot(313);plot(spec_H);

    %%%%save label
    [m,n]=size(spec_Y);
    for i=1:1:m
        for j=1:1:n
            if j==n
                fprintf(label_txt,'%.16f\n',spec_Y(i,j));
            else
                fprintf(label_txt,'%.16f\t',spec_Y(i,j));
            end
        end
    end  
    
    %%%%save peak range matrix
    [m,n]=size(spec_H);
    for i=1:1:m
        for j=1:1:n
            if j==n
                fprintf(range_txt,'%.16f\n',spec_H(i,j));
            else
                fprintf(range_txt,'%.16f\t',spec_H(i,j));
            end
        end
    end  
    
    %%%%save data
    [m,n]=size(spec_X);
    for i=1:1:m
        for j=1:1:n
            if j==n
                fprintf(data_txt,'%.16f\n',spec_X(i,j));
            else
                fprintf(data_txt,'%.16f\t',spec_X(i,j));
            end
        end
    end 
end
fclose(label_txt);
fclose(data_txt);