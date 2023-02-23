clc;clear all;

% ObjectAmplitude=imread('.\train\boat_s.bmp','bmp');
% %ObjectAmplitude=imread('.\train\lena.bmp','bmp');
% ObjectAmplitude=im2double(ObjectAmplitude);
% ObjectAmplitude=rgb2gray(ObjectAmplitude);
% ObjectAmplitude=imresize(ObjectAmplitude,[64*1,128*1]);

tic;
mkdir test;
filename_test_img='.\MNIST_data\t10k-images.idx3-ubyte';
filename_test_lab='.\MNIST_data\t10k-labels.idx1-ubyte';
test_img = loadMNISTImages(filename_test_img); 
test_lab = loadMNISTLabels(filename_test_lab);
num_sum=64;num_test=1;I_max=4;%num_sub=1000;I_max=1;
d=30;length_z=20;
n_layer = 4;                                  %需要分成n个色阶
STEP = 256/n_layer;                            %每个色阶宽度
Theta=0;
theta=Theta/180*pi;
pix=10.8e-3;                        
lam=632.8e-6;
k=2*pi/lam;
n=64;m=64;
[fx,fy]=meshgrid(linspace(-1/2/pix,1/2/pix-1/(m*pix),m),linspace(-1/2/pix,1/2/pix-1/(n*pix),n));

%读入全息图
nameHoloAmp_img_gray=strcat('.\test\momd\','test_output_pred','.mat');
%nameHoloAmp_img_gray=strcat('.\train\','HoloAmp','.mat');
nameHoloAmp_img_struct=load(nameHoloAmp_img_gray);
HoloAmp=double(cell2mat({nameHoloAmp_img_struct.HoloAmp}));

nametrain_input_gray=strcat('.\train\','train_input','.mat');
%nameHoloAmp_img_gray=strcat('.\train\','HoloAmp','.mat');
train_input_struct=load(nametrain_input_gray);
train_input=double(cell2mat({train_input_struct.train_input}));

for num_img=1:num_test
    a=zeros(n,m);
    b=zeros(n,m);
    for i_a_n=1:n
        for i_a_m=1:m
            a(i_a_n,i_a_m)=HoloAmp(num_img,i_a_n,i_a_m);
            b(i_a_n,i_a_m)=train_input(num_img,i_a_n,i_a_m);
        end
    end
    figure(1),imshow (a,[]);
    figure(2),imshow (b,[]);
    for I=4:4
        %引入参考光
        ref=ones(n,m);
        for ii=1:m
            ref(:,ii)=exp(1i*k*ii*tan(theta));
        end
        a=a.*ref;

        %角谱传播
        D=-(d+length_z*I/n_layer);
        %g=exp(1i*k*D*sqrt(1-(lam*fx).^2-(lam*fy).^2));
        g=exp(1i*D*sqrt(k^2-(2*pi*fx).^2-(2*pi*fy).^2));
        af=fftshift(fft2(fftshift(a)));

        %single-sideband filter
        [afn,afm]=size(af);  
        single_filter=ones(afn,afm);                %顺光路，或者以自身的角度来去确定左右，sideband在左则命名为左，如1：afm/2为右，afm/2:afm为左
        for ii=1:afn/2+1
            single_filter(ii,:)=0;
        end
        af=af.*single_filter; 

        e=fftshift(ifft2(fftshift(af.*g)));
        e_abs=abs(e).^2;
        %e_abs=abs(e);
        e_abs = ImageNormalization(e_abs);
        %mse=QualityEvaluationMSE(ObjectAmplitude,e_abs);
        mse=QualityEvaluationMSE(b,e_abs);
        figure(3),imshow (e_abs);
        namee=strcat('.\test\momd\image\','image_pred_',num2str(num_img),'_Depth_',num2str(I),'.bmp');
        %namee=strcat('.\test\momd\image\','image_true_',num2str(num_img),'_Depth_',num2str(I),'.bmp');
        imwrite(e_abs,namee);
    end
    a_gray=mat2gray(a);
    namea=strcat('.\test\momd\','HoloAmp_pred_',num2str(num_img),'.bmp');
    %namea=strcat('.\test\momd\','HoloAmp_true_',num2str(num_img),'.bmp');
    imwrite(a_gray,namea);
end
toc;