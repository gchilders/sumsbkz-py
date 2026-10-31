#include <stdlib.h>
#include <time.h>
#include <fstream.h>
#include <NTL/LLL.h>
#include <NTL/ZZ.h>

#ifdef WIN32
#include <windows.h>
#endif

//#define VERBOSE

NTL_CLIENT

void sortvecs(vec_ZZ&, vec_long&, long);
void hpsortvecs(vec_ZZ&, vec_long&, long);
void randvec(vec_ZZ&, long);
void searchlll(long, long *, long *, long, long, long, long);
void searchlllprime(long, long *, long *, long, long, long, long, long);
void outputsolutions(char *, mat_ZZ&, mat_ZZ&, vec_long&, long, long, long, long *, long *);
long rnd(long);
void CountTotals(const vec_ZZ&, long&, long&);
long OneSideLen(const vec_ZZ&);
long TotalLen(const vec_ZZ&);

long kmin=9;
long kmax=32;
long kprimemin=5;
long kprimemax=32;
long nmin=40;
long nmax=50;

int main(int argc, char *argv[])
{
    
    cout << "SumsBKZ Version 1.3.1 beta" << endl;
    
    srand((unsigned)time(NULL));
    long i, n, blocksize, prune, workterms, primeonly, primemin, includelow;
    workterms=0;
#ifdef WIN32
    long runpriority;
#endif

    if ((argc<4) || 
#ifdef WIN32
        (argc>9) ||
#else
        (argc>8) ||
#endif
        ((n=nmin=nmax=atol(argv[1]))<3) ||
        ((blocksize=atol(argv[2]))<2) ||
        (blocksize>n) ||
        ((prune=atol(argv[3]))<0))
    {
#ifdef WIN32
        cerr << "Command line usage:" << endl;
        cerr << argv[0] << " n blocksize prune [workterms includelow primeonly primemin noidle]" << endl;
#else
        cerr << "Usage:  " << argv[0] << " n blocksize prune [workterms includelow primeonly primemin]" << endl;
#endif
        cerr << "Searches for equal sums of like powers using the BKZ integer" << endl;
        cerr << "relation algorithm." << endl;
        cerr << "Outputs normal solutions to output.txt and prime solutions to poutput.txt." << endl;
        cerr << "n - number of terms in basis.  Minimum of 3." << endl;
        cerr << "blocksize - blocksize for BKZ algorithm between 2 and n inclusive." << endl;
        cerr << "prune - prune for BKZ algorithm.  To disable, use 0." << endl;
        cerr << "workterms - only use the first this many terms of random vector in" << endl;
        cerr << "            reduction.  To disable, leave out or use 0." << endl;
        cerr << "includelow - if workterms is used, ensure elements 1-5 are always included" << endl;
        cerr << "             in the basis." << endl;
        cerr << "primeonly - set to 1 to only work on primes, otherwise 0." << endl;
        cerr << "primemin - set the minimum prime power you want to test or 0 for the default." << endl;
#ifdef WIN32
        cerr << "noidle - if present, the program runs at normal instead of idle priority." << endl;
#endif
        cerr << endl;
        cerr << "Or enter values below:" << endl;
        cerr << "nmin: ";
        cin >> nmin;
        n=nmin;
        cerr << "nmax: ";
        cin >> nmax;
        if (nmin>nmax) {
            cerr << "minimum must be less than maximum" << endl;
            return 1;
        }
        cerr << "blocksize: ";
        cin >> blocksize;
        cerr << "prune: ";
        cin >> prune;
        cerr << "minimum power for normal solutions (0 for no search for normal solutions): ";
        cin >> kmin;
        if (kmin>0) {
            cerr << "maximum power for normal solutions: ";
            cin >> kmax;
        }
        if (kmin>kmax) {
            cerr << "minimum must be less than maximum" << endl;
            return 1;
        }
        if (kmax>100) {
            cerr << "maximum power is 100" << endl;
            kmax=100;
        }
        cerr << "minimum power for prime solutions (0 for no search for prime solutions): ";
        cin >> kprimemin;
        if ((kmin==0)&&(kprimemin==0)) {
            cerr << "No work to do!" << endl;
            return 1;
        }
        if (kprimemin>0) {
            cerr << "maximum power for prime solutions: ";
            cin >> kprimemax;
        }
        if (kprimemin>kprimemax) {
            cerr << "minimum must be less than maximum" << endl;
            return 1;
        }
        if (kprimemax>100) {
            cerr << "maximum power is 100" << endl;
            kprimemax=100;
        }
        cerr << "workterms (0 to disable): ";
        cin >> workterms;
        includelow=0;
#ifdef WIN32
        cerr << "running priority (0=idle, 1=normal): ";
        cin >> runpriority;
#endif
    }
    
#ifdef WIN32
    if ((argc!=9)&&(runpriority!=1)) {
        SetPriorityClass(GetCurrentProcess(),IDLE_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_IDLE);
        cout << "Running with idle priority." << endl;
        cout << endl;
    } else {
        cout << "Running with normal priority." << endl;
    }
#endif
    
    if (argc>4) workterms=atol(argv[4]);
    if ((workterms>n) || (workterms<0)) {
        cerr << "Workterms must be between 0 and n." << endl;
        return 1;
    }
    if (argc>5) includelow=atol(argv[5]); else includelow=0;
    if ((includelow!=0) && (includelow!=1)) {
        cerr << "Includelow must be 0 or 1." << endl;
        return 1;
    }
    if (includelow && !workterms) {
        cerr << "Includelow has no effect if workterms isn't used." << endl;
        includelow=0;
    }
    if (argc>6) primeonly=atol(argv[6]); else primeonly=0;
    if ((primeonly!=0) && (primeonly!=1)) {
        cerr << "Primeonly must be 0 or 1." << endl;
        return 1;
    }
    if (argc>7) primemin=atol(argv[7]); else primemin=0;
    if ((primemin<0) || (primemin>32)) {
        cerr << "Primemin must be between 0 and 32." << endl;
        return 1;
    }
    if (primemin>0) kprimemin=primemin;
    
    cerr << "Using ..." << endl;
    cerr << "nmin: " << nmin << endl;
    cerr << "nmax: " << nmax << endl;
    if (kmin>0) {
        cerr << "kmin: " << kmin << endl;
        cerr << "kmax: " << kmax << endl;
    }
    if (kprimemin>0) {
        cerr << "kprimemin: " << kprimemin << endl;
        cerr << "kprimemax: " << kprimemax << endl;
    }
    cerr << "blocksize: " << blocksize << endl;
    if (prune!=0) cerr << "prune: " << prune << endl;
    if (workterms!=0) cerr << "workterms: " << workterms << endl;
    if (includelow!=0) cerr << "includelow: " << includelow << endl;
    if (primeonly!=0) cerr << "primeonly: " << primeonly << endl;
    if (primemin!=0) cerr << "primemin: " << primemin << endl;
    cerr << endl;
    
    // Output parameters to files...
    if ((!primeonly)&&(kmin>0)) {
        ofstream output("output.txt", ios::app);
        output << "// nmin: " << nmin << endl;
        output << "// nmax: " << nmax << endl;
        output << "// kmin: " << kmin << endl;
        output << "// kmax: " << kmax << endl;
        output << "// blocksize: " << blocksize << endl;
        if (prune!=0) output << "// prune: " << prune << endl;
        if (workterms!=0) output << "// workterms: " << workterms << endl;
        if (includelow!=0) output << "// includelow: " << includelow << endl;
        output << endl;
        output.close();
    }
    
    if (kprimemin>0) {
        ofstream poutput("poutput.txt", ios::app);
        poutput << "// nmin: " << nmin << endl;
        poutput << "// nmax: " << nmax << endl;
        poutput << "// kprimemin: " << kprimemin << endl;
        poutput << "// kprimemax: " << kprimemax << endl;
        poutput << "// blocksize: " << blocksize << endl;
        if (prune!=0) poutput << "// prune: " << prune << endl;
        if (workterms!=0) poutput << "// workterms: " << workterms << endl;
        if (includelow!=0) poutput << "// includelow: " << includelow << endl;
        if (primeonly!=0) poutput << "// primeonly: " << primeonly << endl;
        if (primemin!=0) poutput << "// primemin: " << primemin << endl;
        poutput << endl;
        poutput.close();
    }
    
    long l[101], rl[101], lp[101], rlp[101];
    
    //Limits
    //Normal solutions
    l[9]=9;   rl[9]=4;
    l[10]=11; rl[10]=5;
    l[11]=14; rl[11]=7;
    l[12]=13; rl[12]=6;
    l[13]=16; rl[13]=8;
    l[14]=18; rl[14]=9;
    l[15]=22; rl[15]=11;
    l[16]=22; rl[16]=11;
    l[17]=27; rl[17]=13;
    l[18]=29; rl[18]=14;
    l[19]=32; rl[19]=16;
    l[20]=36; rl[20]=16;
    l[21]=39; rl[21]=16;
    l[22]=43; rl[22]=16;
    l[23]=45; rl[23]=16;
    l[24]=50; rl[24]=16;
    l[25]=54; rl[25]=16;
    l[26]=53; rl[26]=16;
    l[27]=62; rl[27]=16;
    l[28]=65; rl[28]=16;
    l[29]=68; rl[29]=16;
    l[30]=71; rl[30]=16;
    l[31]=76; rl[31]=16;
    l[32]=83; rl[32]=16;
    l[33]=98;
    l[34]=107;
    l[35]=107;
    l[36]=117;
    l[37]=116;
    l[38]=127;
    l[39]=132;
    l[40]=143;
    l[41]=146;
    l[42]=168;
    l[43]=175;
    l[44]=189;
    l[45]=203;
    l[46]=219;
    l[47]=218;
    l[48]=234;
    l[49]=241;
    l[50]=246;
    l[51]=272;
    l[52]=303;
    l[53]=292;
    l[54]=330;
    l[55]=326;
    l[56]=335;
    l[57]=373;
    l[58]=411;
    l[59]=414;
    l[60]=450;
    l[61]=421;
    l[62]=445;
    l[63]=430;
    l[64]=534;
    l[65]=474;
    l[66]=495;
    l[67]=646;
    l[68]=669;
    l[69]=735;
    l[70]=691;
    l[71]=813;
    l[72]=833;
    l[73]=763;
    l[74]=869;
    l[75]=962;
    l[76]=1005;
    l[77]=964;
    l[78]=1122;
    l[79]=1295;
    l[80]=1261;
    l[81]=1289;
    l[82]=1360;
    l[83]=1460;
    l[84]=1698;
    l[85]=1483;
    l[86]=1829;
    l[87]=1729;
    l[88]=2043;
    l[89]=2088;
    l[90]=2176;
    l[91]=2072;
    l[92]=2167;
    l[93]=2634;
    l[94]=2859;
    l[95]=2763;
    l[96]=3057;
    l[97]=3064;
    l[98]=3525;
    l[99]=3578;
    l[100]=6000;
    
    for (i=33;i<=100;i++) rl[i]=16;
    
    //Primes
    lp[5]=7;  rlp[5]=3;
    lp[6]=7;  rlp[6]=3;
    lp[7]=9;  rlp[7]=4;
    lp[8]=13; rlp[8]=6;
    lp[9]=13; rlp[9]=7;
    lp[10]=19;  rlp[10]=9;
    lp[11]=23;  rlp[11]=12;
    lp[12]=23;  rlp[12]=11;
    lp[13]=29;  rlp[13]=15;
    lp[14]=31;  rlp[14]=15;
    lp[15]=38;  rlp[15]=16;
    lp[16]=39;  rlp[16]=16;
    lp[17]=47;  rlp[17]=16;
    lp[18]=45;  rlp[18]=16;
    lp[19]=55;  rlp[19]=16;
    lp[20]=57;  rlp[20]=16;
    lp[21]=67;  rlp[21]=16;
    lp[22]=77;  rlp[22]=16;
    lp[23]=83;  rlp[23]=16;
    lp[24]=81;  rlp[24]=16;
    lp[25]=97;  rlp[25]=16;
    lp[26]=104; rlp[26]=16;
    lp[27]=105; rlp[27]=16;
    lp[28]=117; rlp[28]=16;
    lp[29]=129; rlp[29]=16;
    lp[30]=126; rlp[30]=16;
    lp[31]=147; rlp[31]=16;
    lp[32]=156; rlp[32]=16;
    
    for (i=33;i<=100;i++) {
        lp[i]=6000;
        rlp[i]=16;
    }
    
    for (;;)
    {
        if ((!primeonly)&&(kmin>0)) searchlll(n, l, rl, blocksize, prune, workterms, includelow);
        if (kprimemin>0) searchlllprime(n, lp, rlp, blocksize, prune, workterms, includelow, primemin);
    }
}

long rnd(long n)  // Random number 1..n using high bits of rand()
{
    return (long)((float)n*rand()/(RAND_MAX+1.0)) + 1;
}

void randvec(vec_ZZ& Bas, long n)
{
    vec_ZZ Temp;
    long i, num;
    
    Temp=Bas;
    
    for (i=1;i<=n;i++)
    {
        num=rnd(n);
        while (Temp(num)==-999)
            if (++num>n) num=1;
            Bas(i)=Temp(num);
            Temp(num)=-999;
    }
    
}

void outputsolutions(char* filename, mat_ZZ& Bas, mat_ZZ& B, vec_long& index, long k, long n, long m, long *l, long *rl)
{
    ofstream output(filename, ios::app);
    
    long tot, postot, negtot;
    long i, j, ndplus;
    
    for (i=1;i<=n;i++)
    {
        if(B(i,m)==0)
        {
            CountTotals(B(i),postot,negtot);
            tot=postot+negtot;
            if (postot > negtot)
            {
                swap(postot,negtot);
                B(i)=-B(i);
            }
            if ((tot<=l[k]) || (postot<=rl[k]))
            {
                if (tot<l[k]) l[k]=tot;
                output << "(" << k << "," << postot << "," << tot-postot << ") ";
                ndplus=0;
                for (j=m-1;j>=1;j--)
                {
                    if (B(i,index(j))>0) 
                    {
                        if (ndplus==1) output << "+";
                        ndplus=1;
                        output << Bas(index(j),m);
                        if (B(i,index(j))!=1) output << "*" << B(i,index(j));
                    }
                }
                output << "=";
                ndplus=0;
                for (j=m-1;j>=1;j--)
                {
                    if (B(i,index(j))<0) 
                    {
                        if (ndplus==1) output << "+";
                        ndplus=1;
                        output << Bas(index(j),m);
                        if (B(i,index(j))!=-1) output << "*" << -B(i,index(j));
                    }
                }
                output << endl;
            }
        }
    }
    output.close();
}

void sortvecs(vec_ZZ& Bas, vec_long& index, long n)   
{
    // Simple, basic, reliable sort routine for testing
    long i, search;
    
    for (i=1;i<n;i++)
        for (search=i+1;search<=n;search++)
            if (Bas(search) < Bas(i))
            {
                swap(Bas(search),Bas(i));
                swap(index(search),index(i));
            }
}

void hpsortvecs(vec_ZZ& Bas, vec_long& index, long n)
{
    // Heap sort
    long i,ir,j,l;
    ZZ rrbas;
    long rrb;
    
    if (n<2) return;
    l=(n>>1)+1;
    ir=n;
    
    for (;;) {
        if (l>1) {
            --l;
            rrbas=Bas(l);
            rrb=index(l);
        } else {
            rrbas=Bas(ir);
            rrb=index(ir);
            Bas(ir)=Bas(1);
            index(ir)=index(1);
            if (--ir==1) {
                Bas(1)=rrbas;
                index(1)=rrb;
                break;
            }
        }
        i=l;
        j=l+l;
        while (j<=ir){
            if (j<ir && Bas(j)<Bas(j+1)) j++;
            if (rrbas<Bas(j)) {
                Bas(i)=Bas(j);
                index(i)=index(j);
                i=j;
                j <<= 1;
            } else break;
        }
        Bas(i)=rrbas;
        index(i)=rrb;
    }
}

void CountTotals(const vec_ZZ& V, long& postot, long& negtot)
{
    long i,n;
    n=V.length();
    postot=0;
    negtot=0;
    for (i=1;i<=n;i++) {
        if (V(i)>0) postot+=to_long(V(i)); else negtot-=to_long(V(i));          
    }
}

long OneSideLen(const vec_ZZ& V) 
{   
    long postot, negtot;
    CountTotals(V,postot,negtot);
    return min(postot,negtot);
}

long TotalLen(const vec_ZZ& V)
{
    long postot, negtot;
    CountTotals(V,postot,negtot);
    return postot+negtot;
}

void searchlll(long n, long *l, long *rl, long blocksize, long prune, long workterms, long includelow)
{
    mat_ZZ B, Bas;
    vec_ZZ BasVec;
    long i, j, k, m, ncount, end;
    for (ncount=nmin;ncount<=nmax;ncount++) {
        n=ncount;
        BasVec.SetLength(n);
        
		for (k=kmin;k<=kmax;k++) {
        switch (k)
		{
		case 8:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while ((j%2!=0)&&(j%5!=0)) j++;
				BasVec(i)=j++;
			}
			break;
		case 10:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while ((j%2!=0)&&(j%11!=0)) j++;
				BasVec(i)=j++;
			}
			break;
		case 12:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while ((j%2!=0)&&(j%3!=0)&&(j%7!=0)&&(j%13!=0)) j++;
				BasVec(i)=j++;
			}
			break;
		case 14:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while (j%2!=0) j++;
				BasVec(i)=j++;
			}
			break;
		case 16:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while ((j%2!=0)&&(j%17!=0)) j++;
				BasVec(i)=j++;
			}
			break;
		case 18:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while ((j%3!=0)&&(j%19!=0)) j++;
				BasVec(i)=j++;
			}
			break;
		case 20:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while ((j%2!=0)&&(j%5!=0)&&(j%11!=0)) j++;
				BasVec(i)=j++;
			}
			break;
		case 22:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while (j%23!=0) j++;
				BasVec(i)=j++;
			}
			break;
		case 24:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while ((j%2!=0)&&(j%13!=0)) j++;
				BasVec(i)=j++;
			}
			break;
		case 28:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while ((j%2!=0)&&(j%29!=0)) j++;
				BasVec(i)=j++;
			}
			break;
		case 30:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while (j%31!=0) j++;
				BasVec(i)=j++;
			}
			break;
		case 32:
			BasVec(1) = 1;
			j=2;
			for (i=2;i<=n;i++)
			{
				while ((j%2!=0)&&(j%17!=0)) j++;
				BasVec(i)=j++;
			}
			break;

		default:
		
			if (includelow) {                  //make sure 1..5 is always in basis
				for (i=1;i<=n;i++) BasVec(i)=i+5; //ensure 1..5 is not in basis
				randvec(BasVec,n);                //randomize it
				if (workterms==0) end=n; else end=workterms;
				for (i=1;i<=5;i++) BasVec(end-i+1)=i;   //replace end of basis with 1..5
			} else {                              //do normal stuff
				for (i=1;i<=n;i++) BasVec(i)=i;
				randvec(BasVec,n);
			}
        }
        
		//long t;
        //t=rnd(5)+1;      //try this if workterms > 0 to construct basis.  Didn't really help.
        //for (i=1;i<=n;i++) BasVec(i)=(i-1)*t+rnd(t);
        
        if (workterms>0)
        {
            n=workterms;
            BasVec.SetLength(n);
        }
        
        m=n+1;
        Bas.SetDims(n,m);
        for (i=1;i<=n;i++) {
            for (j=1;j<=m;j++) {
                if (i==j) 
                {
                    Bas(i,j)=1;
                }
                else if (j==m) 
                {
                    Bas(i,j)=BasVec(i);
                }
                else 
                {
                    Bas(i,j)=0;
                }
            }
        }
        
        vec_long index;
        index.SetLength(n);
        for (i=1;i<=n;i++) index(i)=i;
        
        hpsortvecs(BasVec,index,n);
        
        //for (k=kmin;k<=kmax;k++) {
            cout << "k: " << k << "  n: " << ncount << endl;
            B=Bas;
            for (i=1;i<=n;i++)
            {
                power(B(i,m),B(i,m),k);
                B(i,m)*=10000;
            }
            
            if (k>50) LLL_XD(B, 0.99);
            BKZ_FP(B, 0.9999, blocksize, prune);
            //G_BKZ_FP(B, 0.9999, blocksize, prune);
            
            mat_ZZ BB;
            vec_ZZ TempVec;
            long negtot, postot;
            TempVec.SetLength(m);
            
            long nn, norig, nlimit;
            
            nlimit=3*n;
            
            BB.SetDims(nlimit,m);
            
            norig=0;
            for (i=1;i<=n;i++) {
                if (B(i,m)==0) {
                    CountTotals(B(i),postot,negtot);
                    if (postot > negtot)
                    {
                        BB(++norig)=-B(i);
                    } else {
                        BB(++norig)=B(i);
                    }
                }
            }
            nn=norig;
            
            //Optimization routine from Yura
            //Add or subtract each solution as long as it reduces the left hand side
            
            for (i=norig; i>=1; i-- ) {
                //if (TotalLen(BB(i)) > l[k]+15) continue;  //Not sure if this is a good thing
                TempVec=BB(i);
#ifdef VERBOSE
                CountTotals(TempVec,postot,negtot);
                cout << "Left:  (" << k << "," << postot << "," << negtot << ") -> ";
#endif
                bool change = true;
                while ( change ) {
                    change = false;
                    for (j=norig; j>=1; --j ) {
                        if ( i==j ) continue;
                        while ( OneSideLen(TempVec) > OneSideLen(TempVec+BB(j)) ) {
                            TempVec += BB(j);
                            change = true;
                        }
                        while ( OneSideLen(TempVec) > OneSideLen(TempVec-BB(j)) ) {
                            TempVec -= BB(j);
                            change = true;
                        }
                    }
                }
                BB(++nn)=TempVec;
#ifdef VERBOSE
                CountTotals(TempVec,postot,negtot);
                cout << "(" << k << "," << postot << "," << negtot << ")" << endl;
#endif
            }
            
            //Reduce total terms
            for (i=norig; i>=1; i-- ) {
                //if (TotalLen(BB(i)) > l[k]+15) continue;
                TempVec=BB(i);
#ifdef VERBOSE
                CountTotals(TempVec,postot,negtot);
                cout << "Total:  (" << k << "," << postot << "," << negtot << ") -> ";
#endif
                bool change = true;
                while ( change ) {
                    change = false;
                    for (j=norig; j>=1; --j ) {
                        if ( i==j ) continue;
                        while ( TotalLen(TempVec) > TotalLen(TempVec+BB(j)) ) {
                            TempVec += BB(j);
                            change = true;
                        }
                        while ( TotalLen(TempVec) > TotalLen(TempVec-BB(j)) ) {
                            TempVec -= BB(j);
                            change = true;
                        }
                    }
                }
                BB(++nn)=TempVec;
#ifdef VERBOSE
                CountTotals(TempVec,postot,negtot);
                cout << "(" << k << "," << postot << "," << negtot << ")" << endl;
#endif
            }
            
            outputsolutions("output.txt", Bas, BB, index, k, nn, m, l, rl);
           
			//outputsolutions("output.txt", Bas, B, index, k, n, m, l, rl);
        }
    }
}

void searchlllprime(long n, long *lp, long *rlp, long blocksize, long prune, long workterms, long includelow, long primemin)
{
    mat_ZZ B, Bas;
    vec_ZZ BasVec;
    long i, j, k, m, ncount, end;
    PrimeSeq s, ss;
    
    for (ncount=nmin;ncount<=nmax;ncount++) {
        n=ncount;
        BasVec.SetLength(n);
        
        if (includelow) {
            for (i=1;i<=5;i++) s.next();             //eliminate first 5 primes
            for (i=1;i<=n;i++) BasVec(i)=s.next();
            randvec(BasVec,n);
            if (workterms==0) end=n; else end=workterms;
            for (i=1;i<=5;i++) BasVec(end-i+1)=ss.next();  //replace end of basis with first 5 primes
        } else {
            for (i=1;i<=n;i++) BasVec(i)=s.next();
            randvec(BasVec,n);
        }
        
        if (workterms>0) {
            n=workterms;
            BasVec.SetLength(n);
        }
        
        m=n+1;
        Bas.SetDims(n,m);
        for (i=1;i<=n;i++) {
            for (j=1;j<=m;j++) {
                if (i==j) 
                {
                    Bas(i,j)=1;
                }
                else if (j==m) 
                {
                    Bas(i,j)=BasVec(i);
                }
                else 
                {
                    Bas(i,j)=0;
                }
            }
        }
        
        vec_long index;
        index.SetLength(n);
        for (i=1;i<=n;i++) index(i)=i;
        
        hpsortvecs(BasVec,index,n);
        
        if (primemin==0) primemin=5;
        for (k=kprimemin;k<=kprimemax;k++) {
            cout << "kprime: " << k << "  n: " << ncount << endl;
            B=Bas;
            for (i=1;i<=n;i++)
            {
                power(B(i,m),B(i,m),k);
                B(i,m)*=10000;
            }
            
            if (k>50) LLL_XD(B, 0.99);
            BKZ_FP(B, 0.9999, blocksize, prune);
            //G_BKZ_FP(B, 0.9999, blocksize, prune);
            
            mat_ZZ BB;
            vec_ZZ TempVec;
            long negtot, postot;
            TempVec.SetLength(m);
            
            long nn, norig, nlimit;
            
            nlimit=3*n;
            
            BB.SetDims(nlimit,m);
            
            norig=0;
            for (i=1;i<=n;i++) {
                if (B(i,m)==0) {
                    CountTotals(B(i),postot,negtot);
                    if (postot > negtot)
                    {
                        BB(++norig)=-B(i);
                    } else {
                        BB(++norig)=B(i);
                    }
                }
            }
            nn=norig;
            
            //Optimization routine from Yura
            //Add or subtract each solution as long as it reduces the left hand side
            
            for (i=norig; i>=1; i-- ) {
                
                //if (TotalLen(BB(i)) > lp[k]+15) continue;
                TempVec=BB(i);
                
                bool change = true;
                while ( change ) {
                    change = false;
                    for (j=norig; j>=1; --j ) {
                        if ( i==j ) continue;
                        while ( OneSideLen(TempVec) > OneSideLen(TempVec+BB(j)) ) {
                            TempVec += BB(j);
                            change = true;
                        }
                        while ( OneSideLen(TempVec) > OneSideLen(TempVec-BB(j)) ) {
                            TempVec -= BB(j);
                            change = true;
                        }
                    }
                }
                BB(++nn)=TempVec;
            }
            
            //Reduce total terms
            for (i=norig; i>=1; i-- ) {
                //if (TotalLen(BB(i)) > lp[k]+15) continue;
                TempVec=BB(i);
                
                bool change = true;
                while ( change ) {
                    change = false;
                    for (j=norig; j>=1; --j ) {
                        if ( i==j ) continue;
                        while ( TotalLen(TempVec) > TotalLen(TempVec+BB(j)) ) {
                            TempVec += BB(j);
                            change = true;
                        }
                        while ( TotalLen(TempVec) > TotalLen(TempVec-BB(j)) ) {
                            TempVec -= BB(j);
                            change = true;
                        }
                    }
                }
                BB(++nn)=TempVec;
            }
            
            outputsolutions("poutput.txt", Bas, BB, index, k, nn, m, lp, rlp);
        }
    }
}
