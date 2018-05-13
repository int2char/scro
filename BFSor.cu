#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include"pathalg.h"
static const int WORK_SIZE =258;
void BFSor::copydata(int s,vector<edge>&edges,int nodenum){
};
void BFSor::dellocate(){
};
void BFSor::allocate(int maxn,int maxedge){
}
void BFSor::topsort()
{
};
__global__ void cleanb(int *d,int *p,int N,int numoff)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>=N)return;
	d[i+numoff]=100000;
	p[i+numoff]=-1;
};
/*__global__ void cleanb(int *d,int *p,int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>=N)return;
	d[i]=100000;
	p[i]=-1;
};*/
void BFSor::updatE(vector<vector<int>>&tesigns)
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
};
__global__ void Sorb(int *d,int *p,int *sor,int ly,int ye,int yoff,int numoff)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>=ly)return;
	int l=i/ye;
	int id=i%ye;
	int y=sor[id+yoff];
	d[y*ly+l*ye+id+numoff]=0;
};
void BFSor::updatS(vector<vector<Sot>>&stpair)
{
	L[0]=0;
	L[1]=LY1;
	L[2]=LY2;
	S[0]=stpair[0].size();
	S[1]=stpair[1].size();
	stps=stpair;
	ncount=L[1]*S[0]+L[2]*S[1];
	int bigN=ncount*pnodesize;
	int numoff=L[1]*S[0]*pnodesize;
		
	int count=0;
	for(int j=0;j<stpair[0].size();j++)
		sor[count++]=stpair[0][j].s;
	int fs=count;
	for(int j=0;j<stpair[1].size();j++)
		sor[count++]=stpair[1][j].s;
	cudaMemcpy(dev_sor,sor,count*sizeof(int),cudaMemcpyHostToDevice);
	cleanb<<<L[1]*S[0]*pnodesize/512+1,512>>>(dev_d,dev_p,L[1]*S[0]*pnodesize,0);
	cleanb<<<L[2]*S[1]*pnodesize/512+1,512>>>(dev_d,dev_p,L[2]*S[1]*pnodesize,numoff);
	Sorb<<<L[1]*S[0]/512+1,512>>>(dev_d,dev_p,dev_sor,L[1]*S[0],S[0],0,0);
	Sorb<<<L[2]*S[1]/512+1,512>>>(dev_d,dev_p,dev_sor,L[2]*S[1],S[1],fs,numoff);
	Size[0]=nodenum*L[1]*S[0];
	Size[1]=nodenum*L[2]*S[1];
}
void BFSor::init(pair<vector<edge>,vector<vector<int>>>ext,vector<pair<int,int>>stpair,int _nodenum)
{
	//cout<<"in paraller BFS init"<<endl;
	nodenum=_nodenum;
	pnodesize=nodenum;
	edges=ext.first;
	esigns=ext.second;
	stp=stpair;
	W=WD+1;
	//st=new int[edges.size()*LY];
	//te=new int[edges.size()*LY];
	d=new int[nodenum*LY*YE];
	p=new int[nodenum*LY*YE];
	esignes=new int[edges.size()*LY];
	vector<vector<int>>neibn(pnodesize*LY,vector<int>());
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
	sor=new int[2*YE];

	cudaMalloc((void**)&dev_d,YE*LY*nodenum*sizeof(int));
	cudaMalloc((void**)&dev_p,YE*LY*nodenum*sizeof(int));
	cudaMalloc((void**)&dev_mm,(pnodesize+1)*sizeof(int));
	cudaMalloc((void**)&dev_ss,(pnodesize+1)*sizeof(int));
	cudaMalloc((void**)&dev_rudu,edges.size()*sizeof(int));
	cudaMalloc((void**)&dev_rudw,edges.size()*LY*sizeof(int));
	cudaMalloc((void**)&dev_rid,edges.size()*sizeof(int));
	cudaMalloc((void**)&dev_sor,2*YE*sizeof(int));
	cudaMemcpy(dev_rudu,rudu,edges.size()*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rudw,rudw,edges.size()*LY*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rid,rid,edges.size()*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mm,mm,(pnodesize+1)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ss,ss,(pnodesize+1)*sizeof(int),cudaMemcpyHostToDevice);
};
BFSor::BFSor():L(PC+1,0),S(PC,0),NF(PC,0),Size(2,0)
{
};
__global__ void BFSFu(int *rudu,int*rudw,int *rid,int *d,int*p,int K,int EE,int PN,int sizeoff,int leveloff,int yel,int ye,int*mm,int *ss)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>=yel)return;
	int ly=i/ye+leveloff;
	int nid=blockIdx.y;
	int off=sizeoff;
	int ii=nid*yel+i+sizeoff;
	int dm=d[ii];
	for(int k=mm[nid];k<ss[nid]+mm[nid];k++)
		{
			int node=rudu[k];
			if(rudw[k+EE*ly]<0)continue;
			int v=d[off+node*yel+i];
			if(v==K-1&&dm>v){d[ii]=K;break;}
		}
}
__global__ void BFScolor(int *rudu,int*rudw,int *rid,int *d,int*p,int K,int EE,int PN,int sizeoff,int leveloff,int yel,int ye,int*mm,int *ss)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>=yel)return;
	int ly=i/ye+leveloff;
	int nid=blockIdx.y;
	int off=sizeoff;
	int ii=nid*yel+i+sizeoff;
	int dm=d[ii];
	for(int k=mm[nid];k<ss[nid]+mm[nid];k++)
		{
			int node=rudu[k];
			if(rudw[k+EE*ly]<0)continue;
			int v=d[off+node*yel+i];
			if(v+1==d[ii]){p[ii]=rid[k];break;}
		}
}
/*__global__ void BFSfast(int *st,int *te,int *d,int* p,int *stid,int E,int N,int size,int round,int Leveloff,int numoff,int yel,int ye)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>=yel)return;	
	int eid=blockIdx.y;
	int eeid=(i/ye+Leveloff)*E+eid;
	int s=st[eeid],t=te[eeid];
	if(t<0)return;
	int offs=s*yel+numoff;
	int offt=t*yel+numoff;
	if(d[offs+i]==round-1&&d[offt+i]>round)d[offt+i]=round;
}
__global__ void BFScolor(int *st,int *te,int *d,int* p,int *stid,int E,int N,int size,int round,int Leveloff,int numoff,int yel,int ye)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>=yel)return;	
	int eid=blockIdx.y;
	int eeid=(i/ye+Leveloff)*E+eid;
	int s=st[eeid],t=te[eeid];
	if(t<0)return;
	int offs=s*yel+numoff;
	int offt=t*yel+numoff;
	if(d[offs+i]==d[offt+i]-1)p[offt+i]=stid[eeid];
}*/
vector<vector<Rout>> BFSor::routalg(int s,int t,int bw)
{
	//cout<<"blasting "<<endl;
	int kk=1;
	time_t start,end;
	start=clock();
	int size=edges.size()*LY*YE;
	cudaStream_t stream0;
	cudaStreamCreate(&stream0);
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	int leoff=L[1];
	int nuoff=L[1]*S[0]*nodenum;
	dim3 blocks_s1(S[0]*L[1]/512+1,pnodesize);
	dim3 blocks_s2(S[1]*L[2]/512+1,pnodesize);
	int sizeoff=S[0]*L[1]*nodenum;

	for(int i=1;i<WD+1;i++)
		{
			BFSFu<<<blocks_s1,512,0,stream0>>>(dev_rudu,dev_rudw,dev_rid,dev_d,dev_p,i,edges.size(),pnodesize,0,0,S[0]*L[1],S[0],dev_mm,dev_ss);
			BFSFu<<<blocks_s2,512,0,stream1>>>(dev_rudu,dev_rudw,dev_rid,dev_d,dev_p,i,edges.size(),pnodesize,sizeoff,L[1],S[1]*L[2],S[1],dev_mm,dev_ss);
		}
	BFScolor<<<blocks_s1,512,0,stream0>>>(dev_rudu,dev_rudw,dev_rid,dev_d,dev_p,0,edges.size(),pnodesize,0,0,S[0]*L[1],S[0],dev_mm,dev_ss);
	BFScolor<<<blocks_s2,512,0,stream1>>>(dev_rudu,dev_rudw,dev_rid,dev_d,dev_p,0,edges.size(),pnodesize,sizeoff,L[1],S[1]*L[2],S[1],dev_mm,dev_ss);
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream0);
	cudaMemcpy(d,dev_d,ncount*nodenum*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(p,dev_p,ncount*nodenum*sizeof(int),cudaMemcpyDeviceToHost);
	/*for(int i=0;i<1;i++)
	{
		cout<<"********************************************** "<<i<<endl;
		for(int j=3;j<4;j++)
			{
					cout<<endl;
					for(int g=0;g<pnodesize;g++)
						{
							//cout<<p[g*S[0]*L[1]+i*S[0]+j]<<" ";
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
		int teoff=S[y-1]*L[y];
		for(int k=LL[y-1];k<LL[y];k++)
		{	
			if(y==2)offg=sizeoff;
			int boff=(k-LL[y-1])*S[y-1]+offg;
			for(int l=0;l<stps[y-1].size();l++)
			{	int loff=boff+l;
				int s=stps[y-1][l].s;
				vector<int>ters=stps[y-1][l].ters;
				for(int i=0;i<ters.size();i++)
				{
					int id=stps[y-1][l].mmpid[ters[i]];
					int hop=0;
					int tt=ters[i];
					int min=d[tt*teoff+loff];
					if(min>50000)continue;
					int of=loff;
					Rout S(s,tt,id,min,of,k);
					result[y-1].push_back(S);
				}
				count++;
			}
		}
		}
	end=clock();
	//cout<<"GPU time is : "<<end-start<<endl;
	//cout<<"over!"<<endl;
	//cudaFree(dev_te);
	//cudaFree(dev_st);
	//cudaFree(dev_d);
	//cout<<"before return"<<endl;
	return result;
};
/*__global__ void BFSfast(int *st,int *te,int *d,int round,int E,int N,int size)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>size)return;
	int eid=(i%(E*LY));
	int s=st[eid],t=te[eid];
	int off=(i/(E*LY))*N+(eid/E)*N*YE;
	if(d[s+off]==round-1&&d[t+off]>round)
		d[t+off]=round;
}*/
