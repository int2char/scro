#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include"pathalg.h"
static const int WORK_SIZE =258;
void Bellmanor::copydata(int s,vector<edge>&edges,int nodenum){
};
void Bellmanor::dellocate(){
};
void Bellmanor::allocate(int maxn,int maxedge){
}
void Bellmanor::topsort()
{
};
void Bellmanor::updatE(vector<vector<int>>&tesigns)
{
	esigns=tesigns;
	int cou1=0;
	for(int k=0;k<LY;k++)
	{
		for(int i=0;i<pnodesize;i++)
		for(int j=0;j<rus[i].size();j++)
			rudw[cou1++]=esigns[k][ruw[i][j]];
	}
	cudaMemcpy(dev_rudw,rudw,LY*edges.size()*sizeof(int),cudaMemcpyHostToDevice);
}

__global__ void clean(int *d,int *p,int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>=N)return;
	d[i]=100000;
	p[i]=-1;
};
void Bellmanor::updatS(vector<vector<Sot>>&stpair)
{
	L[0]=0;
	L[1]=LY1;
	L[2]=LY2;
	S[0]=stpair[0].size();
	S[1]=stpair[1].size();
	stps=stpair;
	int count=0;
	ncount=L[1]*S[0]+L[2]*S[1];
	int bigN=ncount*nodenum;
	clean<<<bigN/512+1,512,0>>>(dev_d,dev_p,bigN);
	cudaMemcpy(d,dev_d,ncount*nodenum*sizeof(int),cudaMemcpyDeviceToHost);
	for(int k=0;k<L[1];k++)
		{
		for(int j=0;j<stpair[0].size();j++)
			{
			 d[stpair[0][j].s*S[0]*L[1]+k*S[0]+j]=0;
			 count++;
			}
		}
	int off=nodenum*S[0]*L[1];
	for(int k=0;k<L[2];k++)
		{
		for(int j=0;j<stpair[1].size();j++)
			{
			 d[stpair[1][j].s*S[1]*L[2]+k*S[1]+j+off]=0;
			 count++;
			}
		}
	Size[0]=nodenum*L[1]*S[0];
	Size[1]=nodenum*L[2]*S[1];
	cudaMemcpy(dev_d,d,ncount*nodenum*sizeof(int),cudaMemcpyHostToDevice);
}
void Bellmanor::init(pair<vector<edge>,vector<vector<int>>>ext,vector<pair<int,int>>stpair,int _nodenum)
{
	nodenum=_nodenum;
	pnodesize=nodenum/(NUT);
	edges=ext.first;
	esigns=ext.second;
	stp=stpair;
	W=WD+1;
	d=new int[nodenum*LY*YE];
	p=new int[nodenum*LY*YE];
	w=new int[edges.size()*LY];
	esignes=new int[edges.size()*LY];
	vector<vector<int>>nein(pnodesize*LY,vector<int>());
	neibn=nein;
	vector<vector<int>>neie(pnodesize,vector<int>());
	vector<vector<int>>rs(pnodesize,vector<int>());
	vector<vector<int>>rw(pnodesize,vector<int>());
	rus=rs;
	ruw=rw;
	for(int i=0;i<edges.size();i++)
		{
			int s=edges[i].s;
			int t=edges[i].t;
			rus[t].push_back(s);
			ruw[t].push_back(i);
			neibn[s].push_back(t);
			neie[s].push_back(i);
		}
	rudu=new int[edges.size()];
	rudw=new int[edges.size()*LY];
	rid=new int[edges.size()];
	int cou1=0;
	int cou2=0;
	int cou3=0;
	mm=new int[pnodesize+1];
	ss=new int[pnodesize+1];
	int du=0;
	for(int i=0;i<pnodesize;i++)
		{
			ss[i]=rus[i].size();
			mm[i]=du;
			du+=rus[i].size();
			for(int j=0;j<rus[i].size();j++)
				rudu[cou1++]=rus[i][j];
			for(int j=0;j<rus[i].size();j++)	
				rid[cou3++]=ruw[i][j];
		}
	for(int k=0;k<LY;k++)
	{
		for(int i=0;i<pnodesize;i++)
		for(int j=0;j<rus[i].size();j++)
			rudw[cou2++]=esigns[k][ruw[i][j]];
	}
	int count=0;
	cudaMalloc((void**)&dev_d,YE*LY*nodenum*sizeof(int));
	cudaMalloc((void**)&dev_p,YE*LY*nodenum*sizeof(int));
	cudaMalloc((void**)&dev_mm,(pnodesize+1)*sizeof(int));
	cudaMalloc((void**)&dev_ss,(pnodesize+1)*sizeof(int));
	cudaMalloc((void**)&dev_rudu,edges.size()*sizeof(int));
	cudaMalloc((void**)&dev_rudw,edges.size()*LY*sizeof(int));
	cudaMalloc((void**)&dev_rid,edges.size()*sizeof(int));
	cudaMemcpy(dev_rudu,rudu,edges.size()*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rudw,rudw,edges.size()*LY*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rid,rid,edges.size()*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mm,mm,(pnodesize+1)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ss,ss,(pnodesize+1)*sizeof(int),cudaMemcpyHostToDevice);
};
Bellmanor::Bellmanor():L(PC+1,0),S(PC,0),NF(PC,0),Size(2,0)
{
};
__global__ void bellmandu(int *rudu,int*rudw,int *rid,int *d,int*p,int K,int EE,int PN,int sizeoff,int leveloff,int yel,int ye,int*mm,int *ss)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>=yel)return;
	int ly=i/ye+leveloff;
	int nid=blockIdx.y;
	int off=(K-1)*PN*yel+sizeoff;
	int ii=K*PN*yel+nid*yel+i+sizeoff;
	int dm=d[ii];
	int pm=-1;
	for(int k=mm[nid];k<ss[nid]+mm[nid];k++)
		{
			int node=rudu[k];
			if(rudw[k+EE*ly]<0)continue;
			int v=d[off+node*yel+i]+rudw[k+EE*ly];
			if(dm>v)dm=v,pm=rid[k];
		}
	if(d[ii]>dm)
		{
			d[ii]=dm,p[ii]=pm;
		}
}
vector<vector<Rout>> Bellmanor::routalg(int s,int t,int bw)
{
	int kk=1;
	time_t start,end;
	cudaStream_t stream0;
	cudaStreamCreate(&stream0);
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	dim3 blocks_s1(S[0]*L[1]/512+1,pnodesize);
	dim3 blocks_s2(S[1]*L[2]/512+1,pnodesize);
	int sizeoff=S[0]*L[1]*nodenum;
	for(int i=1;i<WD+1;i++)
	{
		bellmandu<<<blocks_s1,512,0,stream0>>>(dev_rudu,dev_rudw,dev_rid,dev_d,dev_p,i,edges.size(),pnodesize,0,0,S[0]*L[1],S[0],dev_mm,dev_ss);
		bellmandu<<<blocks_s2,512,0,stream0>>>(dev_rudu,dev_rudw,dev_rid,dev_d,dev_p,i,edges.size(),pnodesize,sizeoff,L[1],S[1]*L[2],S[1],dev_mm,dev_ss);
	}
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream0);
	cudaMemcpy(d,dev_d,ncount*nodenum*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(p,dev_p,ncount*nodenum*sizeof(int),cudaMemcpyDeviceToHost);
	/*for(int i=39;i<40;i++)
	{
		cout<<"********************************************** "<<i<<endl;
		for(int j=3;j<4;j++)
			{
				cout<<endl;
				for(int k=3;k<4;k++)
					{
						for(int g=33;g<34;g++)
							{
								//cout<<k*S[0]*L[1]*pnodesize+g*S[0]*L[1]+i*S[0]+j<<" ";
								cout<<edges[p[sizeoff+k*S[1]*L[2]*pnodesize+g*S[1]*L[2]+i*S[1]+j]].s<<" ";
								cout<<sizeoff+k*S[1]*L[2]*pnodesize+g*S[1]*L[2]+i*S[1]+j<<endl;
							}
				}
			}
	}*/
	vector<vector<Rout>>result(2,vector<Rout>());
	vector<int>LL(3,0);
	LL=L;
	LL[2]+=LL[1];
	int count=0;
	int offg=0;
	for(int y=1;y<PC+1;y++)
		{
		int leoff=S[y-1]*L[y]*pnodesize;
		int teoff=S[y-1]*L[y];
		for(int k=LL[y-1];k<LL[y];k++)
		{	
			if(y==2)offg=sizeoff;
			int boff=(k-LL[y-1])*S[y-1]+offg;
			for(int l=0;l<stps[y-1].size();l++)
			{	
				int loff=boff+l;
				int s=stps[y-1][l].s;
				vector<int>ters=stps[y-1][l].ters;
				for(int i=0;i<ters.size();i++)
				{
					int id=stps[y-1][l].mmpid[ters[i]];
					int hop=0;
					int tt=ters[i];
					int min=100000;
					int prn=-1;
					for(int v=1;v<W;v++)
						{
						if(d[loff+v*leoff+tt*teoff]<min)
							{	
								min=d[loff+v*leoff+tt*teoff];
								prn=v;
							}
						}
					if(prn<0||min>50000)continue;
					int of=loff+prn*leoff;
					Rout S(s,tt,id,min,of,k);
					result[y-1].push_back(S);
				}
				count++;
			}
		}
		}
	//cout<<"GPU time is : "<<end-start<<endl;
	return result;
};

/*
__global__ void bellmanhigh(int *st,int *te,int *d,int *has,int *w,int E,int N,int size,int *m,int round,int Leveloff,int numoff,int ye,int ly)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>size)return;	
	int eid=(i%(E*ly));
	int eeid=eid+Leveloff;
	int s=st[eeid],t=te[eeid],weight=w[eeid];
	if(weight<0)return;
	int off=(i/(E*ly))*N+(eid/E)*N*ye+numoff;
	//if(has[s+off]<round-1)return;
	if(d[s+off]+weight<d[t+off])  
		{
			d[t+off]=weight+d[s+off];
			//has[t+off]=round;
			*m=1;
		}
}*/
/*__global__ void color(int *st,int *te,int *d,int *pre,int *has,int *w,int E,int N,int size,int round,int Leveloff,int numoff,int ye,int ly)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>size)return;	
	int eid=(i%(E*ly));
	int eeid=eid+Leveloff;
	int s=st[eeid],t=te[eeid],weight=w[eeid];
	if(weight<0)return;
	int off=(i/(E*ly))*N+(eid/E)*N*ye+numoff;
	//if(has[s+off]<round-1)return;
	if(d[s+off]+weight==d[t+off])
		pre[t+off]=s+off;
}*/
/*m1=1;
	*m2=1;
	int round=1;
	cout<<"fuck wx!"<<endl;
	int flag1=0,flag2=0;
	int cc=0;
	while(*m2==1||*m1==1)
	{
		*m2=0,*m1=0;
		cudaMemcpyAsync(dev_m2,m2,sizeof(int),cudaMemcpyHostToDevice,stream1);
		bellmanhigh<<<size[1]/1024+1,1024,0,stream1>>>(dev_st,dev_te,dev_d,dev_has,dev_w,edges.size(),nodenum,size[1],dev_m2,round,leveloff[1],nodeoff[1],S[1],L[1]);
		cudaMemcpyAsync(dev_m1,m1,sizeof(int),cudaMemcpyHostToDevice,stream0);
		bellmanhigh<<<size[0]/1024+1,1024,0,stream0>>>(dev_st,dev_te,dev_d,dev_has,dev_w,edges.size(),nodenum,size[0],dev_m2,round,leveloff[0],nodeoff[0],S[0],L[0]);
		color<<<size[1]/1024+1,1024,0,stream1>>>(dev_st,dev_te,dev_d,dev_p,dev_has,dev_w,edges.size(),nodenum,size[1],round,leveloff[1],nodeoff[1],S[1],L[1]);
		cudaMemcpyAsync(m2,dev_m2,sizeof(int),cudaMemcpyDeviceToHost,stream1);
		color<<<size[0]/1024+1,1024,0,stream0>>>(dev_st,dev_te,dev_d,dev_p,dev_has,dev_w,edges.size(),nodenum,size[0],round,leveloff[0],nodeoff[0],S[0],L[0]);
		cudaMemcpyAsync(m1,dev_m1,sizeof(int),cudaMemcpyDeviceToHost,stream0);
		cudaStreamSynchronize(stream1);
		cudaStreamSynchronize(stream0);
	}*/
