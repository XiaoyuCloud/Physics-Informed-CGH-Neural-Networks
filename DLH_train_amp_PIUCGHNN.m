clc;clear all;
filename_train_img='.\MNIST_data\train-images.idx3-ubyte';
filename_train_lab='.\MNIST_data\train-labels.idx1-ubyte';
train_img = loadMNISTImages(filename_train_img); 
train_lab = loadMNISTLabels(filename_train_lab);
num_sum=64;multi_x=2;multi_y=2;multi=multi_x*multi_y;
num_train=1000;
sum_train=num_train*multi;
d=30;length_z=20;
n = 4;                                  %需要分成n个色阶
STEP = 256/n;                            %每个色阶宽度
Lamda=632.8e-6;                          %波长
dL=10.8e-3;                              %DMD 像素间距
k=2*pi/Lamda;
train_amp_multi=zeros(num_sum,num_sum,multi);
train_amp_posit=zeros(num_sum,num_sum,multi);
train_input=zeros(num_train,num_sum,num_sum);
HoloAmp=zeros(num_train,num_sum,num_sum);
N=num_sum;M=num_sum;
[fx,fy]=meshgrid(linspace(-1/(2*dL),1/(2*dL)-1/(M*dL),M),linspace(-1/(2*dL),1/(2*dL)-1/(N*dL),N));
%设置single-sideband fil ter
single_filter=ones(N,M);
for ii=1:N/2
    single_filter(ii,:)=0;
end
N_sub=28;M_sub=28;
for num_img=1:multi:sum_train
    train_input_amp=zeros(num_sum,num_sum);
    train_input_depth=zeros(num_sum,num_sum);
    for num_sub_x=1: multi_x
        for num_sub_y=1:multi_y
            train_amp_sub=train_img(:,:,num_img+num_sub_x*num_sub_y-1);
            train_amp_sub_binary=train_amp_sub;
            train_amp_sub_binary(train_amp_sub_binary>0)=1;
            %tran_amp_depth_sub=zeros(N_sub,M_sub)+unidrnd(4)*STEP*train_amp_sub_binary;
            tran_amp_depth_sub=zeros(N_sub,M_sub)+4*STEP*train_amp_sub_binary;
            train_amp_sub_posit_x=(num_sum-multi_x*M_sub)/multi_x/2+num_sub_x*M_sub/2+(num_sub_x-1)...
                                   *((num_sum-multi_x*M_sub)/multi_x+M_sub/2);
            train_amp_sub_posit_y=(num_sum-multi_y*N_sub)/multi_y/2+num_sub_y*N_sub/2+(num_sub_y-1)...
                                   *((num_sum-multi_y*N_sub)/multi_y+N_sub/2);
            train_input_amp(train_amp_sub_posit_y-N_sub/2:train_amp_sub_posit_y+N_sub/2-1,...
                            train_amp_sub_posit_x-M_sub/2:train_amp_sub_posit_x+M_sub/2-1)...
                            =train_amp_sub;
            train_input_depth(train_amp_sub_posit_y-N_sub/2:train_amp_sub_posit_y+N_sub/2-1,...
                            train_amp_sub_posit_x-M_sub/2:train_amp_sub_posit_x+M_sub/2-1)...
                            =tran_amp_depth_sub;
        end
    end
train_input_img=train_input_amp;
train_input_img=mat2gray(train_input_img);
num_img_list=(num_img-1)/multi+1;
train_input(num_img_list,:,:)=train_input_img;

AnDiffract=0;
POSITION=train_input_amp;
DEPTH=train_input_depth;
OUTPUT = zeros(N,M,n);
 for I = 1:n
     [x,y]=find(DEPTH>(I-1)*STEP & DEPTH<=I*STEP);
     for J=1:length(x)
         OUTPUT(x(J),y(J),I)=POSITION(x(J),y(J));
     end     
    A=sqrt(OUTPUT(:,:,I));
    %A=im2double(A);
    oz=d+length_z*I/n;
    %A1=A.*exp(1i*2*pi*rand(N,M));
    A1=A;
    HFunct=exp(1i*k*oz.*sqrt(1-(Lamda*fx).^2-(Lamda*fy).^2)); 
    DScreen_F=fftshift(fft2(fftshift(A1))); 
    %DScreen_F=DScreen_F.*single_filter;
    AnDiffract=fftshift(ifft2(fftshift(DScreen_F.*HFunct)))+AnDiffract;
 end


AnF=fftshift(fft2(fftshift(AnDiffract))); 
%figure,imshow(abs(AnF),[]);
AnF=AnF.*single_filter;
%figure,imshow(abs(AnF),[]);
AnDiffract=fftshift(ifft2(fftshift(AnF)));
HoloAmp_img=2*real(AnDiffract);
HoloAmp_img_gray=mat2gray(HoloAmp_img);
HoloAmp(num_img_list,:,:)=HoloAmp_img_gray;
end
path='.\train\';
save ([path 'train_input'],'train_input','-v7.3');
path='.\train\';
save ([path 'HoloAmp'],'HoloAmp','-v7.3');