clc;clear
f=500;lambda=800e-6;k=2*pi/lambda; %[mm]
% I=imread('Near field\Feb25\02_15su_xc4.15mm.bmp');
load 'MyColorMaps.mat';
I11=imread('Sept12/A=-2 B=-+0.55 pol.bmp');
% I=imread('Near field\April26-27\Bessel02_100su.bmp');
% I=imread('Near field\April26-27\ST03_100su.bmp');
I=double(I11);
I_max=(max(max(I))); I=I/I_max;
d=3.7*2.4e-3;%[mm] pixel size of CCD1 - 27BUP031 (input)

s=size(I);
x=(1:s(2))*d - s(2)*d/2 +2.034 ;  y=(1:s(1))*d - s(1)*d/2 +0.22;
% x=(1:s(2))*d -1.35 ;  y=(1:s(1))*d-1.3 ;
kx=2*pi*x/f/lambda;%[rad/um]
ky=2*pi*y/f/lambda;%[rad/um]

h1=figure(10);clf
% imagesc(kx,ky,(I));colormap(hot);

% xlim([-1 1]*0.11);ylim([-1 1]*0.11);
% colormap(blue_red);caxis([-1 1]*0.5)
imagesc(x,y,log(I));colormap(hot);
ylabel('y (mm)');   xlabel('x (mm)'); 
xlim([-1 1]*9);ylim([-1 1]*9    );

h2=figure(11);clf
imagesc(kx,ky,log(I));colormap(hot);
% ylabel('k_y (rad/um)');   xlabel('k_x (rad/um)'); 
xlim([-1 1]*140);ylim([-1 1]*140);
% colormap(blue_red);caxis([-1 1]*0.5)
% imagesc(x,y,log(I));colormap(hot);
ylabel('k_y (rad/mm)');   xlabel('k_x (rad/mm)'); 
% xlim([-1 1]*9);ylim([-1 1]*9    );



% caxis([0 1]*I_max);
% caxis([10 60])
set(h1,'position',[00 50 270 210])
set(h2,'position',[300 50 270 210])
return
%% 1D plot at x=0 and Gaussian fit to it
Iy=I(:,1296);Iy=sgolayfilt(double(Iy),1,13);
Iy=Iy/max(Iy);

%%% Feb25
%%% for 15su
% i1=440;i2=651;
% i3=1212;i4=1451;
%%% 25su
% i3=1070;i4=1280;
%%% March 3; 40 su
% i1=678;i2=957;
% i3=957;i4=1229;
%%% Sept1

y1=y(i1:i2);I1=Iy(i1:i2);
y2=y(i3:i4);I2=Iy(i3:i4);
% 
[sigma,mu1,Amp]=mygaussfit(y1,I1);   
gaus_fit=Amp*exp(-(y1-mu1).^2/(2*sigma^2));
FWHM1=2*sqrt(2*log(2))*sigma
mu1


[sigma2,mu2,Amp]=mygaussfit(y2,I2);   
gaus_fit2=Amp*exp(-(y2-mu2).^2/(2*sigma2^2));
FWHM2=2*sqrt(2*log(2))*sigma2
mu2

figure(11);clf
plot(y,Iy);hold on
plot(y1,gaus_fit);hold on
plot(y2,gaus_fit2);
% xlim([-1 1]*4);
xlabel('y [um]');ylabel('I(y,x=0)');

%% Calculating expected L_max for Bessel beam from G.Indebetouw paper
r=(abs(mu1-mu2))/2%[mm]
dr=(FWHM1+FWHM2)/2%[mm]
% dr=0.37
l=0.8e-3;%[mm]
f=300;%[mm]
L=l*f^2/(r*dr) %[mm]






