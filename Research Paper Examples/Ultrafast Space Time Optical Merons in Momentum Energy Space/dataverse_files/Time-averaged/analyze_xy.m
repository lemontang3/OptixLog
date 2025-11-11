clc;clear

I_H=double(imread(  'A=0.8 B=-6.5\v2\H.bmp'    ));
I_V=double(imread(  'A=0.8 B=-6.5\v2\V.bmp'    ));
I_p45=double(imread('A=0.8 B=-6.5\v2\+45.bmp'  ));
I_m45=double(imread('A=0.8 B=-6.5\v2\-45.bmp'  ));
I_R=double(imread(  'A=0.8 B=-6.5\v2\RCP-v2.bmp'  ));
I_L=double(imread(  'A=0.8 B=-6.5\v2\LCP-v2.bmp'  ));

% I_max=(max(max(I))); I=I/I_max;
d=3.7e-3;%[mm] pixel size of CCD1 - 27BUP031 (input)

s=size(I_H);
x=(1:s(2))*d - s(2)*d/2+0  ;  y=(1:s(1))*d - s(1)*d/2 -0.04;


h1=figure(10);clf
subplot(3,2,1);
imagesc(x+0.04,y,(I_H));colormap(hot);title('H')
ylabel('y (mm)');   xlabel('x (mm)'); 
xlim([-1 1]*0.1);ylim([-1 1]*0.1);
caxis([0 256])

subplot(3,2,2);
imagesc(x,y,(I_V));colormap(hot);title('V')
ylabel('y (mm)');   xlabel('x (mm)'); 
xlim([-1 1]*0.1);ylim([-1 1]*0.1);caxis([0 256])

subplot(3,2,3);
imagesc(x,y,(I_p45));colormap(hot);title('+45')
ylabel('y (mm)');   xlabel('x (mm)'); 
xlim([-1 1]*0.1);ylim([-1 1]*0.1);caxis([0 256])

subplot(3,2,4);
imagesc(x+0.04,y,(I_m45));colormap(hot);title('-45')
ylabel('y (mm)');   xlabel('x (mm)'); 
xlim([-1 1]*0.1);ylim([-1 1]*0.1);caxis([0 256])

subplot(3,2,5);
imagesc(x-0.04,y,(I_R));colormap(hot);title('RCP')
ylabel('y (mm)');   xlabel('x (mm)'); 
xlim([-1 1]*0.1);ylim([-1 1]*0.1);caxis([0 256])

subplot(3,2,6);
imagesc(x,y,(I_L));colormap(hot);title('LCP')
ylabel('y (mm)');   xlabel('x (mm)'); 
xlim([-1 1]*0.1);ylim([-1 1]*0.1);caxis([0 256])





