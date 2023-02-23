Lamda=632.8e-6;                          %波长
dL=10.8e-3;                              %DMD像素间距
k=2*pi/Lamda;

ObjectAmplitude=imread('.\train\boat_s.bmp','bmp');
%ObjectAmplitude=imread('.\train\lena.bmp','bmp');
ObjectAmplitude=im2double(ObjectAmplitude);
ObjectAmplitude=rgb2gray(ObjectAmplitude);
ObjectAmplitude=imresize(ObjectAmplitude,[1080*1,1920*1]);
[N,M]=size(ObjectAmplitude);
[fx,fy]=meshgrid(linspace(-1/(2*dL),1/(2*dL)-1/(M*dL),M),linspace(-1/(2*dL),1/(2*dL)-1/(N*dL),N));
num_train=1;
train_input=zeros(num_train,N,M);
HoloAmp=zeros(num_train,N,M);
train_input(1,:,:)=ObjectAmplitude;

%设置single-sideband filter
single_filter=ones(N,M);
for ii=1:N/2
    single_filter(ii,:)=0;
end
%设置再现像的距离
d=50;AnDiffract=0;
 for I = 1:1
    %A=ObjectAmplitude;
    A=sqrt(ObjectAmplitude);
    oz=d;
    A1=A.*exp(1i*2*pi*rand(N,M));
    %A1=A;
    %HFunct=exp(1i*k*oz.*sqrt(1-(Lamda*fx).^2-(Lamda*fy).^2)); 
    HFunct=exp(1i*oz.*sqrt(k^2-(2*pi*fx).^2-(2*pi*fy).^2)); 
    DScreen_F=fftshift(fft2(fftshift(A1))); 
    AnDiffract=fftshift(ifft2(fftshift(DScreen_F.*HFunct)))+AnDiffract;
 end
AnF=fftshift(fft2(fftshift(AnDiffract))); 
AnF=AnF.*single_filter;
AnDiffract=fftshift(ifft2(fftshift(AnF)));
An=mat2gray(2*real(AnDiffract));
nameAn=strcat('.\train\','holo','.bmp');
imwrite(An,nameAn);

HoloAmp(1,:,:)=An;
imshow(ObjectAmplitude);
path='.\train\';
save ([path 'train_input'],'train_input','-v7.3');
path='.\train\';
save ([path 'HoloAmp'],'HoloAmp','-v7.3');
