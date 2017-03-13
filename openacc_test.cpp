//
//  openacc_test.cpp
//  
//
//  Created by Chen Cai on 3/13/17.
//
//

#include <stdio.h>
#include <iostream>
using namespace std;
#include <cstdlib>
#include <cassert>

int main(int argc, char*argv[])
{
    if (argc <2){
        cerr << "Use:nCount"<<endl;
        return -1;
    }
    
    int nCount=atoi(argv[1])*100000;
    
    if (nCount <0){
        cerr << "ERROR: nCount must be grater than zero!" << endl;
        return -1
    }
    
    // Allocate variables
    char *status = new char[nCount];
    
    // Here is where we fill the status vector
#pragma acc parallel loop copyout(status[0:nCount])
    for(int i=0; i<nCount;i++)
        status[i]=1;
    
    int sum=0;
    for(int i=0; i<nCount;i++)
        sum+=status[i];
    
    cout << "final sum is " << (sum/100000) << "millons" <<endl;
    
    assert(sum==nCount)
    
    delete [] status;
    
    return 0;
    
    
    
}
