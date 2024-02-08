clear
clc
tic
% n_pv=[0	2	3	2	2	1	2	0	2	1	2	2	1	2	0	2	2	2	2	0	2	2	2	3	2	0	2	2	2	2	0	2	2];
% n_ess=[0	1	2	0	1	2	1	0	1	1	2	1	0	1	0	1	3	0	1	1	1	0	1	2	1	3	0	0	1	2	0	1	1];
% u_bs=[0	1	1	0	1	1	1	1	1	1	1	0	1	1	1	0	1	1	1	1	1	1	1	0	1	0	0  0  1  1  1   0   1];
% function [c_ope]=lower_cost(n_pv,n_ess,u_bs)
%% ������������
data_mb=xlsread('Data_input.xlsx','�ڵ���ɾ���');%33*33
data_mg=xlsread('Data_input.xlsx','�ڵ�絼����');%33*33
A=xlsread('Data_input.xlsx','A'); %�ڽӾ���33*33
date_PL=xlsread('Data_input.xlsx','MG��׼�й�����'); %�ڵ���*24
date_QL=xlsread('Data_input.xlsx','MG��׼�޹�����'); %�ڵ���*24
data_fpv=xlsread('Data_input.xlsx','���')/1.5;%�ڵ���*24
data_base=xlsread('Data_input.xlsx','��վ���书��'); %�ڵ���*24
data_price=xlsread('Data_input.xlsx','���'); %ÿ��ʱ�̵�۲�ͬ,1*24
data_utility=xlsread('Data_input.xlsx','Ч�溯��ϵ��');%33*2
b=xlsread('Data_input.xlsx','b');%SNR���Ի�ϵ��
c1=xlsread('Data_input.xlsx','c');%�û�����ĳһ��վ��ϵ��
%����Ͷ�ʳɱ�����
a_pv=0.05; 
a_ess=0.05; 
a_bs=0.05; %������
y_pv=10;   
y_ess=7;   
y_bs=5; %�豸ʹ������
k_pv=(a_pv*(1+a_pv)^y_pv)/((1+a_pv)^y_pv-1);
k_ess=(a_ess*(1+a_ess)^y_ess)/((1+a_ess)^y_ess-1);
k_bs=(a_bs*(1+a_bs)^y_bs)/((1+a_bs)^y_bs-1);%���Ͷ�ʵ�Чϵ��
c_pv=240000;  
c_ess=1000000;  
c_bs=6000000; %��λͶ�ʼ۸�,��λΪ��Ԫ/MW
u_ope=365*(1+0.5*a_pv)/(1+a_pv)^y_pv;%ȫ���ۺ����е�Ч����

mpc=case33bw;%��ȡ����ieee33����
NN=size(mpc.bus,1);%����ڵ���Ŀ
n_branch=size(mpc.branch,1);%����֧·��Ŀ
N=6;%���ڹ���Լ������Բ��Ϊ����
Ntime=24;%ʱ��
Num_lp=NN*NN;%������·��Ŀ
M=100000000;%���޴�ʵ���������ɳۻ�Լ������
s=10;%�������10������
P_emax=110;%��վ���书�ʻ�׼ֵ

P_chamax=5;%1��λ�����豸����繦�ʣ�MW
P_dismax=5;%1��λ�����豸���ŵ繦��,MW
E_batmin=5;%1��λ�����豸��С����,MWh
E_batmax=20;%1��λ�����豸�������,MWh
E_bat0=8;%1��λ�����豸��ʼ����, MWh
e_bat=0.95;%�����豸��ŵ�Ч��
Q_batmax=5;%1��λ�����豸�ų��޹��������ֵ
P_L=date_PL;
Q_L=date_QL;

m3_pv = zeros(11,33);
m3_es = zeros(11,33);
m3_bs = zeros(11,33);
m3_c_total_node = zeros(11,33);
m3_carbon=zeros(11);
for e_bs_i=1:2
    e_rps=-0.1+e_bs_i*0.1;
%% ��ʼ������
Vol_poiac=sdpvar(NN,Ntime,'full');%�ڵ��ѹ��ֵ��ƽ��
Phase_poi=sdpvar(NN,Ntime,'full');%�ڵ��ѹ���
PbqL=sdpvar(Num_lp,Ntime,'full');%����֧·�й����� 
QbqL=sdpvar(Num_lp,Ntime,'full');%����֧·�޹�����
N_pl=sdpvar(NN,Ntime,'full');%���ڵ�����֧·���ʺͣ�����Ϊ��
N_ql=sdpvar(NN,Ntime,'full');%���ڵ�����֧·�޹����ʺͣ�����Ϊ��
Pst_slack=sdpvar(1,Ntime,'full');%ƽ��ڵ��ⲿע���й�����
Pst_slack1=sdpvar(1,Ntime,'full');%ע��������
Pst_slack2=sdpvar(1,Ntime,'full');%ע�븺����
Qst_slack=sdpvar(1,Ntime,'full');%ƽ��ڵ��ⲿע���޹�����
% P_L=sdpvar(NN,Ntime,'full');%���нڵ������й�����  ����ƽ��ڵ�Ϊ�� 
% Q_L=sdpvar(NN,Ntime,'full');%���нڵ������޹�����  ����ƽ��ڵ�Ϊ��
p_loss=sdpvar(2*n_branch,Ntime,'full');

%Ŀ�꺯��
C_L=sdpvar(1,Ntime,'full');%ÿ��ʱ�����нڵ㣨����վ�⣩���õ�ɱ�
U=sdpvar(NN,1,'full');%ÿ���ڵ�����ʱ���Ч�溯��
c=sdpvar(NN,Ntime,'full');%ÿ���ڵ��û�ÿ��ʱ���Ч�溯��
p_line=78;%��·���ϵ��
c_hloss=sdpvar(1,Ntime,'full');
c_in=sdpvar(NN,1,'full');

%�������
P_pv=sdpvar(NN,Ntime,'full');%�����ڵ���ע�빦�ʣ�33*24 
n_pv=intvar(1,NN,'full');

% ����
u_ess=intvar(1,NN,'full');
P_chabat=sdpvar(NN,Ntime,'full');%��繦�� 
P_disbat=sdpvar(NN,Ntime,'full');%�ŵ繦�� 
Q_ibat=sdpvar(NN,Ntime,'full');%���ܷų�����
Index_cha=binvar(NN,Ntime,'full');% ���ָʾ 48
Index_dis=binvar(NN,Ntime,'full');% �ŵ�ָʾ 48
E_bat=sdpvar(NN,Ntime+1,'full'); %��ش��ص����ٷֱ� 
P_bat=sdpvar(NN,Ntime,'full');%�����ڵ㴢��ע�빦��
Q_bat=sdpvar(NN,Ntime,'full');%�ڵ㴢��ע���޹�����

%ͨ�ſɿ���
u_bs=binvar(1,NN,'full');
%N_bs=sdpvar(1,NN,'full');
acc=binvar(NN,NN,'full');%�û������վ������������ʾһ���û�
P_ie=sdpvar(NN,NN,'full');%��������ʾһ���û�
P_e=sdpvar(NN,Ntime,'full');%��վ���书��,��λΪW
SNR_b=sdpvar(NN,1,'full');
P_b=sdpvar(NN,Ntime,'full');%���нڵ��վ�ĵ���
r_b=sdpvar(NN,1,'full');%ͨ�ſɿ���

%% Լ������
Constraints=[];
%% Ͷ������Լ��
n_pv(1)=0;
u_ess(1)=0;
u_bs(1)=0;
% for i=2:NN   
%      Constraints=[Constraints, 0<=n_pv(i)<=40];%����豸��������Լ��
%      Constraints=[Constraints, 0<=u_ess<=100];%�����豸����Լ��
% end
  Constraints=[Constraints, sum(u_bs)>=8 ];%��վ������ĿԼ��
%% ���Լ��
for i=1:NN   
     Constraints=[Constraints, 0<=P_pv(i,:)<=n_pv(i)*data_fpv(i,:)];%�������Լ��   
end
%% ����Լ��
for i=1:NN
    Constraints=[Constraints,  0<=P_chabat(i,:)<=Index_cha(i,:)*M];%�Ƿ���
    Constraints=[Constraints,  0<=P_disbat(i,:)<=Index_dis(i,:)*M];%�Ƿ�ŵ�
    Constraints=[Constraints,  Index_cha(i,:)+Index_dis(i,:)==1];%��ŵ������һ��״̬
    Constraints=[Constraints,   0<=P_chabat(i,:)<=u_ess(i)*P_chamax];%�Ƿ���ڴ���
    Constraints=[Constraints,   0<=P_disbat(i,:)<=u_ess(i)*P_dismax];%�Ƿ���ڴ���
    Constraints=[Constraints,   -Q_batmax*u_ess(i)<=Q_ibat(i,:)<=Q_batmax*u_ess(i)];%�ͷ��޹�����Լ��
    Constraints=[Constraints,   E_bat(i,end)==E_bat0*u_ess(i)];%��ʼ�����ʱ����soc����һ��
    E_bat(i,1)=E_bat0*u_ess(i);%���ܳ�ŵ繫ʽ����һʱ�̴��ܵ���
end
for t=2:(Ntime+1)
    for i=1:NN
        Constraints=[Constraints,   E_bat(i,t)==E_bat(i,t-1)+P_chabat(i,t-1)*e_bat-P_disbat(i,t-1)/e_bat];
     %Constraints=[Constraints,   Soc_bat(i,t)==Soc_bat(i,t-1)+P_chabat(i,t-1)*e_bat/E_batmax-P_disbat(i,t-1)/e_bat/E_batmax];
    end
end
for i=1:NN
    Constraints=[Constraints, 1/6*E_batmax*u_ess(i)<=E_bat(i,t)<=E_batmax*u_ess(i)  ];%��������Լ��
end
  P_bat(:,:)=P_disbat(:,1:Ntime)-P_chabat(:,1:Ntime);%�д��ܽڵ�Լ��  ����ȥ�ŵ磬���ڹ��������ڵ�
  Q_bat(:,:)=Q_ibat(:,1:Ntime);%�д��ܽڵ�Լ��

%% ���Ի�����Լ��
%(12)(13)(14)
for t=1:Ntime
    k=0;
    for i=1:NN%%       
        Cc=A-eye(size(A,1));%��������
        Node_cop=find(Cc(i,:)==1);%�ҵ������ڵ�
        Node_ncop=setxor(1:1:NN,Node_cop);%�������ڵ�
        ceng=NN*(i-1);%������
        for j=1:length(Node_ncop)
         PbqL(ceng+Node_ncop(j),:)=0;  QbqL(ceng+Node_ncop(j),:)=0 ;%%����������
        end
        for j=1:length(Node_cop)
            G=data_mg(i,Node_cop(j));
            B=data_mb(i,Node_cop(j));
            PbqL(ceng+Node_cop(j),t)=G*(Vol_poiac(i,t)-Vol_poiac(Node_cop(j),t))/2-B*(Phase_poi(i,t)-Phase_poi(Node_cop(j),t));
            QbqL(ceng+Node_cop(j),t)=-B*(Vol_poiac(i,t)-Vol_poiac(Node_cop(j),t))/2-G*(Phase_poi(i,t)-Phase_poi(Node_cop(j),t));
% ����ɱ�ҪŪ�ɾ���ֵ
%           p_loss(k+1,t)=PbqL(ceng+Node_cop(j),t)/G*10^6;
%            k=k+1;
        end 
%         for l=k+1:2*n_branch
%             p_loss(l,t)=0;
%         end
        % ���ڵ�����֧·���ʺ�
   
          N_pl(i,t)=sum(PbqL(NN*(i-1)+1:1:NN*i,t))   ; %���ڽڵ�i��tʱ���������ӵ�����֧·�й�����֮��
          N_ql(i,t)=sum(QbqL(NN*(i-1)+1:1:NN*i,t))   ; %���ڽڵ�i��tʱ���������ӵ�����֧·�޹�����֮��
    end
    
end
%���书��Լ��
 Constraints =[Constraints, -0.04<=PbqL<=0.04  -0.04<= QbqL <= 0.04 ]; %��Լ����14���򻯳�����
 
 %% �ڵ��ѹԼ��,���Լ��
 %(15)(16)
 Constraints =[Constraints, -pi/3<=Phase_poi(2:1:NN,:)<pi/3  0.8<= Vol_poiac(2:1:NN,:) <= 1.05 ];%������Ҫ�޸�����
 Constraints =[Constraints, Vol_poiac(1,:)==1.05  Phase_poi(1,:)==0];%��1�ڵ�Ϊ�ο��ڵ㣬���ѹ��ֵ�������һ����
 
 %% �ĵ���Լ��
%  Constraints =[Constraints, 0.8*date_PL<=P_L<=1.2*date_PL ];%ע���ǲ���ͬά�ľ��� ��СֵΪ��׼���ɵ�0.8�������ֵΪ��׼���ɵ�1.2��
%  Constraints =[Constraints, 0.8*date_QL<=Q_L<=1.2*date_QL ];
%  Constraints =[Constraints, sum(date_PL,2)<=sum(P_L,2) ];

%% ����ƽ��Լ��
for t=1:Ntime
    Constraints=[Constraints,  N_pl(1,t)*10000+P_L(1,t)==Pst_slack(1,t)  ];%ѡȡ1�ڵ�Ϊƽ��ڵ� sum(PL_node(:,t))/1000
    Constraints=[Constraints,  N_pl(2:NN,t)*10000+P_L(2:NN,t)-P_pv(2:NN,t)-P_bat(2:NN,t)+P_b(2:NN,t)==0  ];%����Ϊ��
    Constraints=[Constraints,  N_ql(1,t)*10000+Q_L(1,t)==Qst_slack(1,t) ];% ƽ��ڵ�
    Constraints=[Constraints,  N_ql(2:NN,t)*10000+Q_L(2:NN,t)+Q_bat(2:NN,t)==0  ];%   
end
%% ��վ�����û���Լ��
%��ÿ���û���˵
for i=1:NN
    Constraints=[Constraints, acc(:,i)<=u_bs(i)];
    Constraints=[Constraints, P_emax*0.7*acc(:,i)<=P_ie(:,i)<=P_emax*1.2*acc(:,i)];
    Constraints=[Constraints, sum(acc(i,:))==1];
    Constraints=[Constraints, sum(acc(:,i))<=5];
%Constraints=[Constraints,0<=P_ie<=100];
   for t=1:Ntime
       P_e(i,t)=sum(P_ie(:,i));
   end
   Constraints=[Constraints, SNR_b(i,1)==b(1,i)+P_ie(i,:)*c1(:,i)+(u_bs-acc(i,:))*b(2:(NN+1),i)*90];%SNR���Ի����ʽ
   Constraints=[Constraints, SNR_b(i,1)>=2.5*10^6];
end
for i=1:NN
     for t=1:Ntime
         Constraints=[Constraints,P_b(i,t)==(70.22*P_e(i,t)+894.54*u_bs(i))/1000000*1000];%�Ȱѵ�λת����Mw,�ٰ��������300��
     end
end
%��������Դ��
Constraints=[Constraints,  sum(sum(P_pv))>=sum(sum(P_L+P_b))*e_rps];
%(6)(7)(8)
%  %ͨ�ſɿ��Ա��ʽ r_b(SNR_b)
%  Ntime=1;
%  v1=binvar(NN,Ntime);
%  v2=binvar(NN,Ntime);
%  v3=binvar(NN,Ntime);
%  v4=binvar(NN,Ntime);
%  v5=binvar(NN,Ntime);
% t1=sdpvar(NN,Ntime,'full');
% t2=sdpvar(NN,Ntime,'full');
% t3=sdpvar(NN,Ntime,'full');
% t4=sdpvar(NN,Ntime,'full');
% a1=sdpvar(NN,Ntime,'full');
% a2=sdpvar(NN,Ntime,'full');
% a3=sdpvar(NN,Ntime,'full');
% a4=sdpvar(NN,Ntime,'full');
% a5=sdpvar(NN,Ntime,'full');
% a6=sdpvar(NN,Ntime,'full');
% 
%  for t=1:1:Ntime
%     for i=1:1:NN   
%             a1(i,t)=0;
%             a2(i,t)=3.6151*10^(-7)*SNR_b(i,t)-0.36151;
%             a3(i,t)=4.5402*10^(-7)*SNR_b(i,t)-0.5431;
%             a4(i,t)=1.4046*10^(-7)*SNR_b(i,t)+0.3976;
%             a5(i,t)=2.0279*10^(-8)*SNR_b(i,t)+0.8783;
%             a6(i,t)=1;
% Constraints = [Constraints, 
%                          a2(i,t)<=t4(i,t)<=(a2(i,t)+v5(i,t)*M)
%                          a3(i,t)<=t4(i,t)<=(a3(i,t)+(1-v5(i,t))*M)  
%                          a1(i,t)<=t3(i,t)<=(a1(i,t)+v4(i,t)*M)
%                          t4(i,t)<=t3(i,t)<=(t4(i,t)+(1-v4(i,t))*M)
%              (a6(i,t)-v3(i,t)*M)<=t2(i,t)<=a6(i,t)
%          (t3(i,t)-(1-v3(i,t))*M)<=t2(i,t)<=t3(i,t)
%              (a5(i,t)-v2(i,t)*M)<=t1(i,t)<=a5(i,t)
%          (t2(i,t)-(1-v2(i,t))*M)<=t1(i,t)<=t2(i,t)
%              (a4(i,t)-v1(i,t)*M)<=r_b(i,t)<=a4(i,t)
%          (t1(i,t)-(1-v1(i,t))*M)<=r_b(i,t)<=t1(i,t)];
%     end
%  end
%    Constraints=[Constraints, r_b>=0.9];

%% Ŀ�꺯��
for t=1:Ntime
    C_L(t)=Pst_slack(t)*data_price(t)*230;
end
for i=1:NN
    for t=1:Ntime
        c(i,t)=data_utility(i,1)*P_L(i,t)+data_utility(i,2);
    end
    U(i)=sum(c(i,:));%Ч�溯���ļ���
end
c_pur=u_ope*(sum(C_L)-sum(U));

 %Ͷ�ʳɱ�
for i=1:NN
    c_in(i)=n_pv(i)*k_pv*c_pv+u_ess(i)*k_ess*c_ess+u_bs(i)*k_bs*c_bs;
end
c_inv_pv=sum(n_pv*k_pv*c_pv);
c_inv_ess=sum(u_ess*k_ess*c_ess);
c_inv_bs=sum(u_bs*k_bs*c_bs);
c_inv=sum(c_in);
c_total=c_inv+c_pur;
obj=c_total;
%% ���
ops=sdpsettings('solver','cplex','verbose',2);
optimize(Constraints,obj,ops)
%% ��ʾ���
vol=value(Vol_poiac);
vol_r=power(vol,1/2);%��ѹ��ֵ
pha= value(Phase_poi);%��ѹ���
PbqL= value(PbqL);%��·�й�����
Pst_slack=value(Pst_slack);
QbqL=value(QbqL);%��·�޹�����
NPL=value(N_pl*10^4);%���ڵ����߳�����

cope_node=zeros(NN,24);
U=value(U);
c_in=value(c_in);
for i=1:NN
    for t=1:Ntime
        cope_node(i,t)=-NPL(i,t)*data_price(t)*230;
    end
    c_pur_node(i)=u_ope*(sum(cope_node(i,:))-U(i));
end
c_total_node=c_pur_node+c_in';

%���
PV_poi=value(P_pv);%�����ϵͳ���͹���
n_pv=value(n_pv);
total_pv=sum(n_pv);
%����
u_ess=value(u_ess);
P_iESS=value(P_chabat(:,1:Ntime)-P_disbat(:,1:Ntime));
P_bat=value(P_bat);%�����й���ŵ�
Q_bat=value(Q_bat);%�����޹���ŵ�
E_bat=value(E_bat);%���״̬
P_chabat1=value(P_chabat);
P_disbat1=value(P_disbat);
total_ess=sum(u_ess)*20;

% MG�ĵ���
P_L=value(P_L);%MGʵ�������й�
Q_L=value(Q_L);%MGʵ�������޹�
%��վ
u_bs=value(u_bs);
%N_bs=value(N_bs);
acc=value(acc);
P_e=value(P_e);%��վ���书��,��λΪW
P_b=value(P_b);%���нڵ��վ�ĵ���
SNR_b=value(SNR_b);%ÿ���û������վ�������
r_b=value(r_b);%ͨ�ſɿ���
%���
C_L=value(C_L);
c=value(c);
U=value(U);
obj=value(obj);
c_inv_pv=value(c_inv_pv);
c_inv_ess=value(c_inv_ess);
c_inv_bs=value(c_inv_bs);
c_inv=value(c_inv);
c_pur=value(c_pur);
for i=1:NN
r_b(i)=power(1-qfunc(power(SNR_b(i)/1000*0.002898,0.5)),125);%ͨ�ſɿ���
r_downlink(i)=20*log2(1+SNR_b(i));%���д�������
end
D_total=sum(r_downlink);

m3_pv(e_bs_i,:)=n_pv;
m3_es(e_bs_i,:)=u_ess*20;
m3_bs(e_bs_i,:)=u_bs;
m3_c_total_node(e_bs_i,:)=c_total_node;
m3_carbon(e_bs_i)=sum(C_L)*0.583;
end
m3_pv2=sum(m3_pv,2);
m3_es2=sum(m3_es,2);
m3_bs2=sum(m3_bs,2);
m3_c_total=sum(m3_c_total_node,2);
toc