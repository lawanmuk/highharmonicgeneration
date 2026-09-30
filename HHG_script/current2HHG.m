function current2HHG( filename,V, offset )

 if ~exist('offset','var')
     % third parameter does not exist, so default it to something
      offset = 0;
      if ~exist('V','var')
          V = 1;
      end
 end

au2fs = 0.024189;

tcurrent = importdata(filename);
current = tcurrent.data;
clear tcurrent;

%current(:,2) = current(:,2)/au2fs;


mask = ones(length(current),1);
l=50;
mask(end-l:end) = (cos((0:l)./l*pi/2)).^2;
% lmin = 0;%2600*4;
% if(lmin>0)
% mask(1:lmin+1) = 0;
% mask(lmin+1:lmin+l+1) = (sin((0:l)./l*pi/2)).^2;
% end

currentx = current(:,3+offset)/V;
currentx = currentx.*mask;
[En,HHGx] = Integration_trap(current(:,2),currentx, 100, 0.01);

currenty = current(:,4+offset)/V;
currenty = currenty.*mask;
[En,HHGy] = Integration_trap(current(:,2),currenty, 100, 0.01);

currentz = current(:,5+offset)/V;
currentz = currentz.*mask;
[En,HHGz] = Integration_trap(current(:,2),currentz, 100, 0.01);

HHG = HHGx + HHGy + HHGz;

%Time derivative
HHG2 = HHG.*En.^2;
HHG2x = HHGx.*En.^2;
HHG2y = HHGy.*En.^2;
HHG2z = HHGz.*En.^2;

if(offset==0)
  dlmwrite(['HHG_' filename '.out'], [En' HHG' HHG2' HHG2x' HHG2y' HHG2z' ], 'delimiter', '\t');
end
if(offset==3)
  dlmwrite(['HHG_' filename '_In.out'], [En' HHG' HHG2' HHG2x' HHG2y' HHG2z' ], 'delimiter', '\t');
end
if(offset==6)
   dlmwrite(['HHG_' filename '_sp1.out'], [En' HHG' HHG2' HHG2x' HHG2y' HHG2z' ], 'delimiter', '\t');
end
if(offset==9)
   dlmwrite(['HHG_' filename '_sp2.out'], [En' HHG' HHG2' HHG2x' HHG2y' HHG2z' ], 'delimiter', '\t');
end
%hold on
%figure
semilogy(En/0.8266, HHG.*En.^2);
end
