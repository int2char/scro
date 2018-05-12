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
__global__ void cleanb(int *d,int *p,int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i>=N)return;
	d[i]=100000;
	p[i]=-1;
};
void BFSor::updatE(vector<vector<int>>&esigns)
{
	int count=0;
	for(int k=0;k<LY;k++)
		for(int i=0;i<nodenum;i++)
			for(int j=0;j<nein[i].size();j++)
				{
					if(esigns[k][neie[i][j]]<0)
						te[count]=-1;
					else
						te[count]=nein[i][j];
					count++;
			}
	cudaMemcpy(dev_te,te,LY*edges.size()*sizeof(int),cudaMemcpyHostToDevice);
};
void BFSor::updatS(vector<vector<Sot>>&stpair)
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
	cleanb<<<bigN/512+1,512,0>>>(dev_d,dev_p,bigN);
	cudaMemcpy(d,dev_d,ncount*nodenum*sizeof(int),cudaMemcpyDeviceToHost);
	for(int k=0;k<L[1];k++)
		{
		for(int j=0;j<stpair[0].size();j++)
			{
			 d[(stpair[0][j].s)*L[1]*S[0]+k*S[0]+j]=0;
			 count++;
			}
		}
	int numoff=count*nodenum;
	for(int k=0;k<L[2];k++)
		{
		for(int j=0;j<stpair[1].size();j++)
			{
			 d[(stpair[1][j].s)*L[2]*S[1]+k*S[1]+j+numoff]=0;
			 count++;
			}
		}
	Size[0]=edges.size()*L[1]*S[0];
	Size[1]=edges.size()*L[2]*S[1];
	cudaMemcpy(dev_d,d,ncount*nodenum*sizeof(int),cudaMemcpyHostToDevice);
}
void BFSor::init(pair<vector<edge>,vector<vector<int>>>ext,vector<pair<int,int>>stpair,int _nodenum)
{
	//cout<<"in paraller BFS init"<<endl;
	nodenum=_nodenum;
	edges=ext.first;
	vector<vector<int>>esigns;
	esigns=ext.second;
	stp=stpair;
	mark=new int;
	*mark=0;
	W=WD+1;
	st=new int[edges.size()*LY];
	te=new int[edges.size()*LY];
	stid=new int[edges.size()*LY];
	d=new int[nodenum*LY*YE];
	p=new int[nodenum*LY*YE];
	esignes=new int[edges.size()*LY];
	vector<vector<int>>ein(nodenum*LY,vector<int>());
	neibn=ein;
	vector<vector<int>>eie(nodenum,vector<int>());
	neie=eie;
	for(int i=0;i<edges.size();i++)
		{
			int s=edges[i].s;
			int t=edges[i].t;
			neibn[s].push_back(t);
			neie[s].push_back(i);
		}
	nein=neibn;
	int count=0;
	for(int k=0;k<LY;k++)
		for(int i=0;i<nodenum;i++)
			for(int j=0;j<neibn[i].size();j++)
					{
						st[count]=i;
						if(esigns[k][neie[i][j]]<0)
							te[count]=-1;
						else
						te[count]=neibn[i][j];
						stid[count]=neie[i][j];
						count++;
					}
	for(int i=0;i<nodenum*LY*YE;i++)
		d[i]=2*WD+1,p[i]=-1;
	cudaMalloc((void**)&dev_st,LY*edges.size()*sizeof(int));
	cudaMalloc((void**)&dev_te,LY*edges.size()*sizeof(int));
	cudaMalloc((void**)&dev_stid,LY*edges.size()*sizeof(int));
	cudaMalloc((void**)&dev_d,YE*LY*nodenum*sizeof(int));
	cudaMalloc((void**)&dev_p,YE*LY*nodenum*sizeof(int));
	cudaMemcpy(dev_te,te,LY*edges.size()*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_st,st,LY*edges.size()*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_stid,stid,LY*edges.size()*sizeof(int),cudaMemcpyHostToDevice);
};
BFSor::BFSor():L(PC+1,0),S(PC,0),NF(PC,0),Size(2,0)
{
};
__global__ void BFSfast(int *st,int *te,int *d,int* p,int *stid,int E,int N,int size,int round,int Leveloff,int numoff,int yel,int ye)
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
}
vector<vector<Rout>> BFSor::routalg(int s,int t,int bw)
{
	cout<<"blasting "<<endl;
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
	dim3 blocks_s1(S[0]*L[1]/512+1,edges.size());
	dim3 blocks_s2(S[1]*L[2]/512+1,edges.size());
	for(int i=1;i<WD+1;i++)
		{
			BFSfast<<<blocks_s1,512,0,stream0>>>(dev_st,dev_te,dev_d,dev_p,dev_stid,edges.size(),nodenum,Size[0],i,0,0,S[0]*L[1],S[0]);
			BFSfast<<<blocks_s2,512,0,stream1>>>(dev_st,dev_te,dev_d,dev_p,dev_stid,edges.size(),nodenum,Size[1],i,leoff,nuoff,S[1]*L[2],S[1]);
		}
	BFScolor<<<blocks_s1,512,0,stream0>>>(dev_st,dev_te,dev_d,dev_p,dev_stid,edges.size(),nodenum,Size[0],0,0,0,S[0]*L[1],S[0]);
	BFScolor<<<blocks_s2,512,0,stream1>>>(dev_st,dev_te,dev_d,dev_p,dev_stid,edges.size(),nodenum,Size[1],0,leoff,nuoff,S[1]*L[2],S[1]);
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream0);
	cudaMemcpy(d,dev_d,ncount*nodenum*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(p,dev_p,ncount*nodenum*sizeof(int),cudaMemcpyDeviceToHost);
	/*for(int i=8;i<9;i++)
	{
		cout<<endl;
		cout<<"ascsacaaaaaaaaaaaaaaaaaaaaaaaa: "<<i<<endl;
		for(int j=0;j<1;j++)
			{
				cout<<endl;
				for(int k=79;k<80;k++)
					{
						cout<<p[i*S[1]*L[2]+j*S[1]+k+nuoff]<<" ";
						cout<<d[i*S[1]*L[2]+j*S[1]+k+nuoff]<<" ";
						cout<<i*S[1]*L[2]+j*S[1]+k+nuoff<<endl;
						//cout<<i<<" "<<S[1]<<" "<<L[2]<<" "<<offg<<" "<<(k-LL[y-1])<<" "<<S[y-1]<<" "<<l<<endl;
					}
			}
	}*/
	vector<vector<Rout>>result(2,vector<Rout>());
	int offer=L[1]*nodenum*stps[0].size();
	vector<int>LL(3,0);
	LL=L;
	LL[2]+=LL[1];
	int count=0;
	int offg=0;
	for(int y=1;y<PC+1;y++)
		for(int k=LL[y-1];k<LL[y];k++)
		{	
			if(y==2)offg=nuoff;
			for(int l=0;l<stps[y-1].size();l++)
			{	
				int s=stps[y-1][l].s;
				vector<int>ters=stps[y-1][l].ters;
				for(int i=0;i<ters.size();i++)
				{
					int id=stps[y-1][l].mmpid[ters[i]];
					int t=ters[i];
					int ds=d[t*S[y-1]*L[y]+offg+(k-LL[y-1])*S[y-1]+l];
					if(ds>WD)continue;
					Rout S(s,t,id,ds,offg+(k-LL[y-1])*stps[y-1].size()+l,k);
					result[y-1].push_back(S);			
				}
				count++;
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
