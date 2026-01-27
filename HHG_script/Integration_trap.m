function [ En,z, Re, Im ] = Integration_trap( t, javt, maxE, dE )

 if ~exist('maxE','var')
     % third parameter does not exist, so default it to something
      maxE = 50;
 end

 if ~exist('dE','var')
     % third parameter does not exist, so default it to something
      dE = 0.01;
 end

au2fs = 0.024189;
au2eV = 27.2113845;

dt = t(2)-t(1);
estep = min(4.13566733/((max(t))*au2fs)/au2eV,dE/au2eV);

Emax = 4.13566733/(dt*au2fs)/au2eV/2;
Emax = min(Emax,maxE/au2eV);
Nenergy = ceil(Emax/estep);
NTiter = length(javt);

z = zeros(1, Nenergy);
En = zeros(1, Nenergy);
Re = zeros(1, Nenergy);
Im = zeros(1, Nenergy);

tmp = (0:NTiter-1)'*dt;

for ie=0:Nenergy-1
    e=ie*estep;
    
    etmp_sin=dt*sum(sin(tmp*e).*javt);
    etmp_sin=etmp_sin-dt*sin((NTiter-1)*e*dt)*javt(NTiter)/2;
    
    etmp_cos=dt*sum(cos(tmp*e).*javt);
    etmp_cos=etmp_cos-dt*javt(1)/2;
    etmp_cos=etmp_cos-dt*cos((NTiter-1)*e*dt)*javt(NTiter)/2;
     
    Re(ie+1) = etmp_cos;
    Im(ie+1) = -etmp_sin;
    z(ie+1) = etmp_sin^2+etmp_cos^2;
    En(ie+1) = au2eV*e;
end
end

