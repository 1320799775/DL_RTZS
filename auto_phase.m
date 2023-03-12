function theta=auto_phase(spec,mode)
    N1=120;
    N2=120;
    spec0=spec/max(abs(spec));
    fn=length(spec0);
%     temp=zeros(1,100);
    niu=2;
    dw=pi/fn/N2/niu; % the biggest phase is pi/(2*niu)
    if mode==1
        for k=1:N1
            theta=2*pi/N1*(k-1);
            spec_temp=spec0.*exp(i*theta);
            spec_temp=real(spec_temp);
%             temp(k)=sum( (spec_temp(spec_temp>0)).^2 );  % KEY PART: used to determine terminal condition (maximize sum(x^2), x>0)
            sum_pos=sum( (spec_temp(spec_temp>0)).^2 );
            sum_neg=sum( (spec_temp(spec_temp<0)).^2 );
            temp(k)=sum_pos/(sum_neg+1e-6);
%             temp(k)=sum_pos;
%             temp(k)=sum( (spec_temp>0).^2 );
%             temp(k)=sum(spec_temp>0); % original judging criterion
        end
        [maxi kk]=max(temp);
        theta=2*pi/N1*(kk-1);
%         disp(kk);
%         disp(temp(kk));
    else if mode==2
        for k=1:N1
            for n=1:N2
%                 theta=2*pi/N1*(k-1)-([0:(fn-1)]-round(fn/2))*dw*(n-1);
                theta=2*pi/N1*(k-1)-linspace(1,0,fn)*2*pi/N2*(n-1);
                spec_temp=spec0.*exp(i*theta);
                spec_temp=real(spec_temp);
%                 temp(k,n)=sum( (spec_temp(spec_temp>0)).^2 );
%                 temp(k,n)=sum( (spec_temp(spec_temp>0)).^2 )/sum( (spec_temp(spec_temp<0)).^2 );
                sum_pos=sum( (spec_temp(spec_temp>0)).^2 );
                sum_neg=sum( (spec_temp(spec_temp<0)).^2 );
                temp(k,n)=sum_pos/(sum_neg+1e-10);
            end
        end
        [maxi kk]=max(temp);
        [maxii kn]=max(maxi);
        theta=2*pi/N1*(kk(kn)-1)-linspace(1,0,fn)*2*pi/N2*(kn-1);
%         disp(kk(kn));disp(kn);
%         disp(temp(kk(kn),kn));
    end
    end
end