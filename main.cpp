#include <iostream>
#include"Graph.h"
#include"pathalg.h"
#include<sys/time.h>
int main(int args,char*arg[])
{
	ofstream outfile;
	outfile.open("data.txt", ios::app);
	cout<<"*******************************************************************************"<<endl;
	cout<<NODE<<" "<<LY<<" "<<DSIZE*40<<" "<<SERT<<" "<<IFHOP<<endl;
	srand(1);
	if(IFHOP>0)
	{
		Bellmanor d1=Bellmanor();
		PBellmanor d2=PBellmanor();
		ERGraph g(NODE,1,d2,d1);	
		double sert=SERT;
		double lambda=1/sert;
		switch (arg[1][0])
		  {
			  case 'F':
			  {
					  cout<<"parall relaize:"<<endl;
					  g.run(0.02,lambda,OBNUM,0,1);
					  break;
			  }
			  case 'P':
			  {
					  cout<<"p relaize:"<<endl;
					  g.run(0.02,lambda,OBNUM,0,0);
					  break;
			  }
			  case 'S':
			  {
					  cout<<"s relaize:"<<endl;
					  g.run(0.02,lambda,OBNUM,1,0);
					  break;
			  }
			  case 'G':
			  {
					  cout<<"g relaize:"<<endl;
					  g.run(0.02,lambda,OBNUM,2,0);
					  break;
			  }
			  
		  }
	}
	else
	{
		BFSor d1=BFSor();
		PBFSor d2=PBFSor();
		ERGraph g(NODE,1,d2,d1);	
		double sert=SERT;
		double lambda=1/sert;
		switch (arg[1][0])
		  {
			  case 'F':
			  {
					  cout<<"parall relaize:"<<endl;
					  g.run(0.02,lambda,OBNUM,0,1);
					  break;
			  }
			  case 'P':
			  {
					  cout<<"p relaize:"<<endl;
					  g.run(0.02,lambda,OBNUM,0,0);
					  break;
			  }
			  case 'S':
			  {
					  cout<<"s relaize:"<<endl;
					  g.run(0.02,lambda,OBNUM,1,0);
					  break;
			  }
			  case 'G':
			  {
					  cout<<"g relaize:"<<endl;
					  g.run(0.02,lambda,OBNUM,2,0);
					  break;
			  }
			  
		  }
	}
	cout<<"*******************************************************************************"<<endl;
	cout<<endl;
	cout<<endl;
}
