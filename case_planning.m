clear
clc
tic
% n_pv=[0	2	3	2	2	1	2	0	2	1	2	2	1	2	0	2	2	2	2	0	2	2	2	3	2	0	2	2	2	2	0	2	2];
% n_ess=[0	1	2	0	1	2	1	0	1	1	2	1	0	1	0	1	3	0	1	1	1	0	1	2	1	3	0	0	1	2	0	1	1];
% u_bs=[0	1	1	0	1	1	1	1	1	1	1	0	1	1	1	0	1	1	1	1	1	1	1	0	1	0	0  0  1  1  1   0   1];
% function [c_ope]=lower_cost(n_pv,n_ess,u_bs)
%% 输入网架数据
data_mb=xlsread('Data_input.xlsx','节点电纳矩阵');%33*33
data_mg=xlsread('Data_input.xlsx','节点电导矩阵');%33*33
A=xlsread('Data_input.xlsx','A'); %邻接矩阵33*33
date_PL=xlsread('Data_input.xlsx','MG标准有功功率'); %节点数*24
date_QL=xlsread('Data_input.xlsx','MG标准无功功率'); %节点数*24
data_fpv=xlsread('Data_input.xlsx','光伏')/1.5;%节点数*24
data_base=xlsread('Data_input.xlsx','基站发射功率'); %节点数*24
data_price=xlsread('Data_input.xlsx','电价'); %每个时刻电价不同,1*24
data_utility=xlsread('Data_input.xlsx','效益函数系数');%33*2
b=xlsread('Data_input.xlsx','b');%SNR线性化系数
c1=xlsread('Data_input.xlsx','c');%用户接入某一基站的系数
%输入投资成本数据
a_pv=0.05; 
a_ess=0.05; 
a_bs=0.05; %贴现率
y_pv=10;   
y_ess=7;   
y_bs=5; %设备使用年限
k_pv=(a_pv*(1+a_pv)^y_pv)/((1+a_pv)^y_pv-1);
k_ess=(a_ess*(1+a_ess)^y_ess)/((1+a_ess)^y_ess-1);
k_bs=(a_bs*(1+a_bs)^y_bs)/((1+a_bs)^y_bs-1);%年度投资等效系数
c_pv=240000;  
c_ess=1000000;  
c_bs=6000000; %单位投资价格,单位为美元/MW
u_ope=365*(1+0.5*a_pv)/(1+a_pv)^y_pv;%全年折合运行等效因子

mpc=case33bw;%读取配网ieee33数据
NN=size(mpc.bus,1);%计算节点数目
n_branch=size(mpc.branch,1);%计算支路数目
N=6;%用于功率约束，将圆分为几段
Ntime=24;%时间
Num_lp=NN*NN;%可能线路数目
M=100000000;%无限大实数，用于松驰或约束变量
s=10;%随机生成10个场景
P_emax=110;%基站发射功率基准值

P_chamax=5;%1单位储能设备最大充电功率，MW
P_dismax=5;%1单位储能设备最大放电功率,MW
E_batmin=5;%1单位储能设备最小容量,MWh
E_batmax=20;%1单位储能设备最大容量,MWh
E_bat0=8;%1单位储能设备初始容量, MWh
e_bat=0.95;%储能设备充放电效率
Q_batmax=5;%1单位储能设备放出无功功率最大值
P_L=date_PL;
Q_L=date_QL;

m3_pv = zeros(11,33);
m3_es = zeros(11,33);
m3_bs = zeros(11,33);
m3_c_total_node = zeros(11,33);
m3_carbon=zeros(11);
for e_bs_i=1:2
    e_rps=-0.1+e_bs_i*0.1;
%% 初始化变量
Vol_poiac=sdpvar(NN,Ntime,'full');%节点电压幅值的平方
Phase_poi=sdpvar(NN,Ntime,'full');%节点电压相角
PbqL=sdpvar(Num_lp,Ntime,'full');%交流支路有功功率 
QbqL=sdpvar(Num_lp,Ntime,'full');%交流支路无功功率
N_pl=sdpvar(NN,Ntime,'full');%各节点相连支路功率和，流出为正
N_ql=sdpvar(NN,Ntime,'full');%各节点相连支路无功功率和，流出为正
Pst_slack=sdpvar(1,Ntime,'full');%平衡节点外部注入有功功率
Pst_slack1=sdpvar(1,Ntime,'full');%注入正功率
Pst_slack2=sdpvar(1,Ntime,'full');%注入负功率
Qst_slack=sdpvar(1,Ntime,'full');%平衡节点外部注入无功功率
% P_L=sdpvar(NN,Ntime,'full');%所有节点消耗有功功率  对于平衡节点为零 
% Q_L=sdpvar(NN,Ntime,'full');%所有节点消耗无功功率  对于平衡节点为零
p_loss=sdpvar(2*n_branch,Ntime,'full');

%目标函数
C_L=sdpvar(1,Ntime,'full');%每个时间所有节点（除基站外）的用电成本
U=sdpvar(NN,1,'full');%每个节点所有时间的效益函数
c=sdpvar(NN,Ntime,'full');%每个节点用户每个时间的效益函数
p_line=78;%线路损耗系数
c_hloss=sdpvar(1,Ntime,'full');
c_in=sdpvar(NN,1,'full');

%光伏变量
P_pv=sdpvar(NN,Ntime,'full');%交流节点光伏注入功率，33*24 
n_pv=intvar(1,NN,'full');

% 储能
u_ess=intvar(1,NN,'full');
P_chabat=sdpvar(NN,Ntime,'full');%充电功率 
P_disbat=sdpvar(NN,Ntime,'full');%放电功率 
Q_ibat=sdpvar(NN,Ntime,'full');%储能放出功率
Index_cha=binvar(NN,Ntime,'full');% 充电指示 48
Index_dis=binvar(NN,Ntime,'full');% 放电指示 48
E_bat=sdpvar(NN,Ntime+1,'full'); %电池储藏电量百分比 
P_bat=sdpvar(NN,Ntime,'full');%交流节点储能注入功率
Q_bat=sdpvar(NN,Ntime,'full');%节点储能注入无功功率

%通信可靠性
u_bs=binvar(1,NN,'full');
%N_bs=sdpvar(1,NN,'full');
acc=binvar(NN,NN,'full');%用户接入基站矩阵，行向量表示一个用户
P_ie=sdpvar(NN,NN,'full');%行向量表示一个用户
P_e=sdpvar(NN,Ntime,'full');%基站发射功率,单位为W
SNR_b=sdpvar(NN,1,'full');
P_b=sdpvar(NN,Ntime,'full');%所有节点基站耗电量
r_b=sdpvar(NN,1,'full');%通信可靠性

%% 约束条件
Constraints=[];
%% 投资数量约束
n_pv(1)=0;
u_ess(1)=0;
u_bs(1)=0;
% for i=2:NN   
%      Constraints=[Constraints, 0<=n_pv(i)<=40];%光伏设备建设数量约束
%      Constraints=[Constraints, 0<=u_ess<=100];%储能设备数量约束
% end
  Constraints=[Constraints, sum(u_bs)>=8 ];%基站建设数目约束
%% 光伏约束
for i=1:NN   
     Constraints=[Constraints, 0<=P_pv(i,:)<=n_pv(i)*data_fpv(i,:)];%光伏功率约束   
end
%% 储能约束
for i=1:NN
    Constraints=[Constraints,  0<=P_chabat(i,:)<=Index_cha(i,:)*M];%是否充电
    Constraints=[Constraints,  0<=P_disbat(i,:)<=Index_dis(i,:)*M];%是否放电
    Constraints=[Constraints,  Index_cha(i,:)+Index_dis(i,:)==1];%充放电仅存在一种状态
    Constraints=[Constraints,   0<=P_chabat(i,:)<=u_ess(i)*P_chamax];%是否存在储能
    Constraints=[Constraints,   0<=P_disbat(i,:)<=u_ess(i)*P_dismax];%是否存在储能
    Constraints=[Constraints,   -Q_batmax*u_ess(i)<=Q_ibat(i,:)<=Q_batmax*u_ess(i)];%释放无功功率约束
    Constraints=[Constraints,   E_bat(i,end)==E_bat0*u_ess(i)];%初始与结束时储能soc保持一致
    E_bat(i,1)=E_bat0*u_ess(i);%储能充放电公式，第一时刻储能电量
end
for t=2:(Ntime+1)
    for i=1:NN
        Constraints=[Constraints,   E_bat(i,t)==E_bat(i,t-1)+P_chabat(i,t-1)*e_bat-P_disbat(i,t-1)/e_bat];
     %Constraints=[Constraints,   Soc_bat(i,t)==Soc_bat(i,t-1)+P_chabat(i,t-1)*e_bat/E_batmax-P_disbat(i,t-1)/e_bat/E_batmax];
    end
end
for i=1:NN
    Constraints=[Constraints, 1/6*E_batmax*u_ess(i)<=E_bat(i,t)<=E_batmax*u_ess(i)  ];%储能容量约束
end
  P_bat(:,:)=P_disbat(:,1:Ntime)-P_chabat(:,1:Ntime);%有储能节点约束  充电减去放电，属于功率流出节点
  Q_bat(:,:)=Q_ibat(:,1:Ntime);%有储能节点约束

%% 线性化潮流约束
%(12)(13)(14)
for t=1:Ntime
    k=0;
    for i=1:NN%%       
        Cc=A-eye(size(A,1));%上三角阵
        Node_cop=find(Cc(i,:)==1);%找到相连节点
        Node_ncop=setxor(1:1:NN,Node_cop);%非相连节点
        ceng=NN*(i-1);%索引层
        for j=1:length(Node_ncop)
         PbqL(ceng+Node_ncop(j),:)=0;  QbqL(ceng+Node_ncop(j),:)=0 ;%%索引无问题
        end
        for j=1:length(Node_cop)
            G=data_mg(i,Node_cop(j));
            B=data_mb(i,Node_cop(j));
            PbqL(ceng+Node_cop(j),t)=G*(Vol_poiac(i,t)-Vol_poiac(Node_cop(j),t))/2-B*(Phase_poi(i,t)-Phase_poi(Node_cop(j),t));
            QbqL(ceng+Node_cop(j),t)=-B*(Vol_poiac(i,t)-Vol_poiac(Node_cop(j),t))/2-G*(Phase_poi(i,t)-Phase_poi(Node_cop(j),t));
% 线损成本要弄成绝对值
%           p_loss(k+1,t)=PbqL(ceng+Node_cop(j),t)/G*10^6;
%            k=k+1;
        end 
%         for l=k+1:2*n_branch
%             p_loss(l,t)=0;
%         end
        % 各节点相连支路功率和
   
          N_pl(i,t)=sum(PbqL(NN*(i-1)+1:1:NN*i,t))   ; %对于节点i在t时刻与其连接的所有支路有功功率之和
          N_ql(i,t)=sum(QbqL(NN*(i-1)+1:1:NN*i,t))   ; %对于节点i在t时刻与其连接的所有支路无功功率之和
    end
    
end
%传输功率约束
 Constraints =[Constraints, -0.04<=PbqL<=0.04  -0.04<= QbqL <= 0.04 ]; %将约束（14）简化成这样
 
 %% 节点电压约束,相角约束
 %(15)(16)
 Constraints =[Constraints, -pi/3<=Phase_poi(2:1:NN,:)<pi/3  0.8<= Vol_poiac(2:1:NN,:) <= 1.05 ];%可能需要修改数据
 Constraints =[Constraints, Vol_poiac(1,:)==1.05  Phase_poi(1,:)==0];%以1节点为参考节点，其电压幅值和相角是一定的
 
 %% 耗电量约束
%  Constraints =[Constraints, 0.8*date_PL<=P_L<=1.2*date_PL ];%注意是不是同维的矩阵 最小值为标准负荷的0.8倍，最大值为标准负荷的1.2倍
%  Constraints =[Constraints, 0.8*date_QL<=Q_L<=1.2*date_QL ];
%  Constraints =[Constraints, sum(date_PL,2)<=sum(P_L,2) ];

%% 功率平衡约束
for t=1:Ntime
    Constraints=[Constraints,  N_pl(1,t)*10000+P_L(1,t)==Pst_slack(1,t)  ];%选取1节点为平衡节点 sum(PL_node(:,t))/1000
    Constraints=[Constraints,  N_pl(2:NN,t)*10000+P_L(2:NN,t)-P_pv(2:NN,t)-P_bat(2:NN,t)+P_b(2:NN,t)==0  ];%流出为正
    Constraints=[Constraints,  N_ql(1,t)*10000+Q_L(1,t)==Qst_slack(1,t) ];% 平衡节点
    Constraints=[Constraints,  N_ql(2:NN,t)*10000+Q_L(2:NN,t)+Q_bat(2:NN,t)==0  ];%   
end
%% 基站接入用户数约束
%对每个用户来说
for i=1:NN
    Constraints=[Constraints, acc(:,i)<=u_bs(i)];
    Constraints=[Constraints, P_emax*0.7*acc(:,i)<=P_ie(:,i)<=P_emax*1.2*acc(:,i)];
    Constraints=[Constraints, sum(acc(i,:))==1];
    Constraints=[Constraints, sum(acc(:,i))<=5];
%Constraints=[Constraints,0<=P_ie<=100];
   for t=1:Ntime
       P_e(i,t)=sum(P_ie(:,i));
   end
   Constraints=[Constraints, SNR_b(i,1)==b(1,i)+P_ie(i,:)*c1(:,i)+(u_bs-acc(i,:))*b(2:(NN+1),i)*90];%SNR线性化表达式
   Constraints=[Constraints, SNR_b(i,1)>=2.5*10^6];
end
for i=1:NN
     for t=1:Ntime
         Constraints=[Constraints,P_b(i,t)==(70.22*P_e(i,t)+894.54*u_bs(i))/1000000*1000];%先把单位转换成Mw,再把整体提高300倍
     end
end
%可再生能源比
Constraints=[Constraints,  sum(sum(P_pv))>=sum(sum(P_L+P_b))*e_rps];
%(6)(7)(8)
%  %通信可靠性表达式 r_b(SNR_b)
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

%% 目标函数
for t=1:Ntime
    C_L(t)=Pst_slack(t)*data_price(t)*230;
end
for i=1:NN
    for t=1:Ntime
        c(i,t)=data_utility(i,1)*P_L(i,t)+data_utility(i,2);
    end
    U(i)=sum(c(i,:));%效益函数的计算
end
c_pur=u_ope*(sum(C_L)-sum(U));

 %投资成本
for i=1:NN
    c_in(i)=n_pv(i)*k_pv*c_pv+u_ess(i)*k_ess*c_ess+u_bs(i)*k_bs*c_bs;
end
c_inv_pv=sum(n_pv*k_pv*c_pv);
c_inv_ess=sum(u_ess*k_ess*c_ess);
c_inv_bs=sum(u_bs*k_bs*c_bs);
c_inv=sum(c_in);
c_total=c_inv+c_pur;
obj=c_total;
%% 求解
ops=sdpsettings('solver','cplex','verbose',2);
optimize(Constraints,obj,ops)
%% 显示结果
vol=value(Vol_poiac);
vol_r=power(vol,1/2);%电压幅值
pha= value(Phase_poi);%电压相角
PbqL= value(PbqL);%线路有功潮流
Pst_slack=value(Pst_slack);
QbqL=value(QbqL);%线路无功潮流
NPL=value(N_pl*10^4);%各节点连线潮流和

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

%光伏
PV_poi=value(P_pv);%光伏向系统输送功率
n_pv=value(n_pv);
total_pv=sum(n_pv);
%储能
u_ess=value(u_ess);
P_iESS=value(P_chabat(:,1:Ntime)-P_disbat(:,1:Ntime));
P_bat=value(P_bat);%储能有功充放电
Q_bat=value(Q_bat);%储能无功充放电
E_bat=value(E_bat);%电池状态
P_chabat1=value(P_chabat);
P_disbat1=value(P_disbat);
total_ess=sum(u_ess)*20;

% MG耗电量
P_L=value(P_L);%MG实际消耗有功
Q_L=value(Q_L);%MG实际消耗无功
%基站
u_bs=value(u_bs);
%N_bs=value(N_bs);
acc=value(acc);
P_e=value(P_e);%基站发射功率,单位为W
P_b=value(P_b);%所有节点基站耗电量
SNR_b=value(SNR_b);%每个用户接入基站的信噪比
r_b=value(r_b);%通信可靠性
%结果
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
r_b(i)=power(1-qfunc(power(SNR_b(i)/1000*0.002898,0.5)),125);%通信可靠性
r_downlink(i)=20*log2(1+SNR_b(i));%下行传输速率
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