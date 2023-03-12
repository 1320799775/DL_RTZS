/* 
Pulse sequence of 1D real-time pure shift experiments using band-selective/Zangger-Sterk
Peter Kiraly

----------------------
Developed By NMR Group
School of Chemistry
University of Manchester
United Kingdom
Aug 2017
----------------------
*/


#include <standard.h>


static int	ph1[64] = {0,2, 0,2, 0,2,0,2, 1,3,1,3,1,3,1,3, 0,2,0,2,0,2,0,2,1,3,1,3,1,3,1,3, 0,2,0,2,0,2,0,2,1,3,1,3,1,3,1,3, 0,2,0,2,0,2,0,2,1,3,1,3,1,3,1,3},
		ph2[64] = {0,0, 0,0, 1,1,1,1, 0,0,0,0,1,1,1,1, 0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1, 2,2,2,2,3,3,3,3,2,2,2,2,3,3,3,3, 2,2,2,2,3,3,3,3,2,2,2,2,3,3,3,3},
		ph3[64] = {0,0, 1,1, 0,0,1,1, 0,0,1,1,0,0,1,1, 2,2,3,3,2,2,3,3,2,2,3,3,2,2,3,3, 0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1, 2,2,3,3,2,2,3,3,2,2,3,3,2,2,3,3},
		ph4[64] = {0,0, 1,1, 0,0,1,1, 0,0,1,1,0,0,1,1, 2,2,3,3,2,2,3,3,2,2,3,3,2,2,3,3, 0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1, 2,2,3,3,2,2,3,3,2,2,3,3,2,2,3,3},
		ph5[64] = {0,2, 2,0, 2,0,0,2, 1,3,3,1,3,1,1,3, 0,2,2,0,2,0,0,2,1,3,3,1,3,1,1,3, 0,2,2,0,2,0,0,2,1,3,3,1,3,1,1,3, 0,2,2,0,2,0,0,2,1,3,3,1,3,1,1,3};
static int	  ph6_m4[ 4] = {0,0,2,2},
		  ph6_m8[ 8] = {0,0,2,2, 0,2,2,0},
		  ph6_t5[ 5] = {0,5,2,5,0},
		  ph6_t7[ 7] = {0,7,20,17,20,7,0},
		  ph6_t9[ 9] = {0,1,12,11,18,11,12,1,0},
		 ph6_m16[16] = {0,0,2,2, 0,2,2,0, 2,2,0,0, 2,0,0,2},
		 ph6_m32[32] = {0,0,2,2, 2,0,0,2, 2,2,0,0, 0,2,2,0,  0,0,0,2, 2,2,0,0, 2,2,2,0, 0,0,2,2},
		 ph6_m64[64] = {0,0,2,2, 2,0,0,2, 2,2,0,0, 0,2,2,0,  0,0,0,2, 2,2,0,0, 2,2,2,0, 0,0,2,2,
				2,2,0,0, 0,2,2,0, 0,0,2,2, 2,0,0,2,  2,2,2,0, 0,0,2,2, 0,0,0,2, 2,2,0,0},
		ph6_t5m4[20] = {0,5,2,5,0, 0,5,2,5,0, 6,11,8,11,6, 6,11,8,11,6},
	       ph6_t5m16[80] = {0,5,2,5,0, 0,5,2,5,0, 6,11,8,11,6, 6,11,8,11,6,
				0,5,2,5,0, 6,11,8,11,6, 6,11,8,11,6, 0,5,2,5,0,
				6,11,8,11,6, 6,11,8,11,6, 0,5,2,5,0, 0,5,2,5,0,
				6,11,8,11,6, 0,5,2,5,0, 0,5,2,5,0, 6,11,8,11,6},
	        ph6_t7m4[28] = {0,7,20,17,20,7,0, 0,7,20,17,20,7,0, 12,19,32,29,32,19,12, 12,19,32,29,32,19,12},
	       ph6_t7m16[112]= {0,7,20,17,20,7,0, 0,7,20,17,20,7,0, 12,19,32,29,32,19,12, 12,19,32,29,32,19,12,
				0,7,20,17,20,7,0, 12,19,32,29,32,19,12, 12,19,32,29,32,19,12, 0,7,20,17,20,7,0,
				12,19,32,29,32,19,12, 12,19,32,29,32,19,12, 0,7,20,17,20,7,0, 0,7,20,17,20,7,0,
				12,19,32,29,32,19,12, 0,7,20,17,20,7,0, 0,7,20,17,20,7,0, 12,19,32,29,32,19,12,},
		ph6_t9m4[36] = {0,1,12,11,18,11,12,1,0, 0,1,12,11,18,11,12,1,0, 12,13,24,23,30,23,24,13,12, 12,13,24,23,30,23,24,13,12},
		ph6[ 4] = {0,0,0,0};	

pulsesequence()
{	
double	rof3 = getval("rof3"),
	tau_a = getval("tau_a"),
	tau_p = getval("tau_p"),
	tau_r = 0.0,
	tpwr = getval("tpwr"),
	tpwrf = getval("tpwrf"),
	pw = getval("pw"),
	pp = 2.0*pw, //getval("pp"),
	pplvl = tpwr, //getval("pplvl"),
	pplvlf = getval("pplvlf"),
	pw180_a = getval("pw180_a"),
	pwr180_a = getval("pwr180_a"),
	hsgt = getval("hsgt"),
	hsglvl = getval("hsglvl"),
	gstab = getval("gstab"),
	gt1 = getval("gt1"),  
	gzlvl1 = getval("gzlvl1"),
	gt2 = getval("gt2"),  
	gzlvl2 = getval("gzlvl2"),
	gzlvl7 = getval("gzlvl7"),
	kp_pfgtc = getval("kp_pfgtc"),  
	ACQ_gt1 = getval("ACQ_gt1"),  
	ACQ_gzlvl1 = getval("ACQ_gzlvl1"),
	ACQ_gzlvl2 = getval("ACQ_gzlvl2"),
	ACQ_gzlvl3 = getval("ACQ_gzlvl3"),
	ACQ_gzlvl4 = getval("ACQ_gzlvl4"),
	ACQ_gstab = getval("ACQ_gstab"),
	kp_npoints = getval("kp_npoints"),
	droppts1 = getval("droppts1"),	
	droppts2 = getval("droppts2"),	
	kp_cycles =(double) (floor)( getval("kp_cycles") -1.0 );
	F_initval(kp_cycles,v10);	
int 	kpph = getval("kpph"),	
	kp_scyc_len;

char	sspul[MAXSTR],
	lkgate_flg[MAXSTR],
	kp_scyc[MAXSTR],
	shp_a[MAXSTR];
  
	getstr("sspul",sspul);
	getstr("lkgate_flg",lkgate_flg);
	getstr("kp_scyc",kp_scyc);
	getstr("shp_a",shp_a);	

if (gzlvl7==0.0) kp_pfgtc=0.0;	
	tau_r = (tau_p - 4.0*ACQ_gt1-4.0*ACQ_gstab-pw180_a -kp_pfgtc-6.0*rof1-pp )/2.0; 
setacqmode(WACQ|NZ);	
if ((kpph<0) || (kpph>64))
{
abort_message("Number of steps for phase cycling is incorrect..change kpph..");
}
if ( (0.25*kp_npoints/sw-gt1-gstab) < 0.0)
{
abort_message("chunktime is too short to accomodate gt1/gstab..");
}
if ((tau_a-rof1-gt2-gstab-2.0*kp_pfgtc)<0.0)
{
abort_message("tau_a is too short to accomodate gt2/gstab..");
}
if ( fmod( np/(2.0*(droppts1+droppts2+kp_npoints)),1.0) != 0.0)
{
abort_message("np must be an integer multiple of (droppts1+kp_npoints+droppts2)..");
}
if (kpph==0)
{
	settable(t1,64,ph1);
	settable(t2,64,ph2);
	settable(t3,64,ph3);
	settable(t4,64,ph4);
	settable(t5,64,ph5);
}
else
{
	settable(t1,kpph,ph1);
	settable(t2,kpph,ph2);
	settable(t3,kpph,ph3);
	settable(t4,kpph,ph4);
	settable(t5,kpph,ph5);
}
	getelem(t1, ct, v1);
	getelem(t2, ct, v2);
	getelem(t3, ct, v3);
	getelem(t4, ct, v4);
	getelem(t5, ct, oph);

kp_scyc_len=strlen(kp_scyc);
if (kp_scyc[0]=='n')
{
	settable(t6,4,ph6);
}
	if (kp_scyc_len==2) 
	{
if ((kp_scyc[0]=='m') && (kp_scyc[1]=='4'))
{
settable(t6,4,ph6_m4);
}
if ((kp_scyc[0]=='m') && (kp_scyc[1]=='8'))
{
settable(t6,8,ph6_m8);
}
if ((kp_scyc[0]=='t') && (kp_scyc[1]=='5'))
{
settable(t6,5,ph6_t5);
obsstepsize(30.0);
}
if ((kp_scyc[0]=='t') && (kp_scyc[1]=='7'))
{
settable(t6,7,ph6_t7);
obsstepsize(15.0);
}
if ((kp_scyc[0]=='t') && (kp_scyc[1]=='9'))
{
settable(t6,9,ph6_t9);
obsstepsize(15.0);
}
	}
	if (kp_scyc_len==3) 
	{
if ((kp_scyc[0]=='m') && (kp_scyc[1]=='1'))
{
settable(t6,16,ph6_m16);
}
if ((kp_scyc[0]=='m') && (kp_scyc[1]=='3'))
{
settable(t6,32,ph6_m32);
}
if ((kp_scyc[0]=='m') && (kp_scyc[1]=='6'))
{
settable(t6,64,ph6_m64);
}
	}
	if (kp_scyc_len==4) 
	{
if (kp_scyc[1]=='5')
{
settable(t6,20,ph6_t5m4);
obsstepsize(30.0);
}
if (kp_scyc[1]=='7')
{
settable(t6,28,ph6_t7m4);
obsstepsize(15.0);
}
if (kp_scyc[1]=='9')
{
settable(t6,36,ph6_t9m4);
obsstepsize(15.0);
}
	}
	if (kp_scyc_len==5) 
	{
if (kp_scyc[1]=='5')
{
settable(t6,80,ph6_t5m16);
obsstepsize(30.0);
}
if (kp_scyc[1]=='7')
{
settable(t6,112,ph6_t7m16);
obsstepsize(15.0);
}
	}

	txphase(zero);
	xmtrphase(zero);
        obsoffset(tof);
	obspower(tpwr);
	obspwrf(tpwrf);
	delay(0.001);

	if (sspul[0]=='y') 
	{
	if ( (lkgate_flg[0] == 'y') || (lkgate_flg[0] == 'k') )  lk_hold(); 
	delay(0.001);
	zgradpulse(hsglvl,hsgt); 
	delay(gstab);
	rgpulse(pw,zero,rof1,rof1);
	zgradpulse(hsglvl,hsgt); 
	delay(0.05);
	}
	if ( (lkgate_flg[0] == 'y') || (lkgate_flg[0] == 'k') )  lk_sample();
	delay(d1);
	if ( (lkgate_flg[0] == 'y') || (lkgate_flg[0] == 'k') )  lk_hold(); 
	delay(0.001);
status(B);
	rgpulse(pw,v1,rof1,0.0);
	obspower(pplvl);	
	if (tpwrf!=pplvlf) obspwrf(pplvlf);
	delay(0.25*kp_npoints/sw-rof1-gt1-gstab -50.0e-9);
	zgradpulse(gzlvl1,gt1);
	delay(gstab);
	rgpulse(pp,v2,rof1,rof1);
	obspower(pwr180_a); 
	if (tpwrf!=pplvlf) obspwrf(4095.0);
	if ( (0.25*kp_npoints/sw-rof1-gt1-gstab  -50.0e-9) > gstab )
	{
	delay(gstab);
	zgradpulse(gzlvl1,gt1);
	delay(0.25*kp_npoints/sw-rof1-gt1-gstab  -50.0e-9);
	}
	else
	{
	delay(0.25*kp_npoints/sw-rof1-gt1-gstab  -50.0e-9);
	zgradpulse(gzlvl1,gt1);
	delay(gstab);
	}
	delay(droppts1/sw);
	delay(tau_a-rof1-gt2-gstab-2.0*kp_pfgtc +rof2);
	zgradpulse(gzlvl2,gt2);
	delay(gstab);
	if (gzlvl7>0.0)
	{
	rgradient('z',gzlvl7);
	delay(kp_pfgtc);
	}
	shaped_pulse(shp_a,pw180_a,v3,rof1,rof1);
	if (gzlvl7>0.0)
	{
	rgradient('z',0.0);
	delay(kp_pfgtc);
	}
	if ((tau_a-rof1-gt2-gstab) > gstab)
	{
        delay(gstab);
	zgradpulse(gzlvl2,gt2);
	delay(tau_a-rof1-gt2-gstab);
	}
	else
	{
	delay(tau_a-rof1-gt2-gstab);
	zgradpulse(gzlvl2,gt2);
        delay(gstab);
	}
status(C);
	startacq(getval("alfa"));

loop(v10,v11);
	sample((droppts1+kp_npoints+droppts2)/sw);
	recoff();	
	obspower(pplvl);
	if (tpwrf!=pplvlf) obspwrf(pplvlf);
	obsunblank();
	getelem(t6,v11,v7);
	if (kp_scyc[0]=='t')
	{
	xmtrphase(v7);
	}
	else
	{
	add(v7,v4,v4);
	}
	mod4(v11,v12);
	delay(rof1-50.0e-9);
	ifzero(v12);
	 zgradpulse(1.0*ACQ_gzlvl1,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v12);
	ifrtEQ(v12,one,v13);
	 zgradpulse(1.0*ACQ_gzlvl2,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	ifrtEQ(v12,two,v13);
	 zgradpulse(1.0*ACQ_gzlvl3,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	ifrtEQ(v12,three,v13);
	 zgradpulse(1.0*ACQ_gzlvl4,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	rgpulse(pp,v4,rof1,rof1);
	delay(rof1);
	ifzero(v12);
	 zgradpulse(1.0*ACQ_gzlvl1,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v12);
	ifrtEQ(v12,one,v13);
	 zgradpulse(1.0*ACQ_gzlvl2,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	ifrtEQ(v12,two,v13);
	 zgradpulse(1.0*ACQ_gzlvl3,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	ifrtEQ(v12,three,v13);
	 zgradpulse(1.0*ACQ_gzlvl4,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	obspower(pwr180_a);
	delay((droppts1+droppts2)/sw);
	delay(tau_r-50.0e-9);
	ifzero(v12);
	 zgradpulse(1.0*ACQ_gzlvl3,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v12);
	ifrtEQ(v12,one,v13);
	 zgradpulse(1.0*ACQ_gzlvl4,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	ifrtEQ(v12,two,v13);
	 zgradpulse(1.0*ACQ_gzlvl1,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	ifrtEQ(v12,three,v13);
	 zgradpulse(1.0*ACQ_gzlvl2,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	if (gzlvl7>0.0)
	{
	rgradient('z',gzlvl7);
	delay(kp_pfgtc);
	}
	shaped_pulse(shp_a,pw180_a,v4,rof1,rof1);
	if (gzlvl7>0.0)
	{
	rgradient('z',0.0);
	delay(kp_pfgtc);
	}
	if (ACQ_gstab>tau_r)
	{
	ifzero(v12);
	 delay(tau_r-rof3);
	 zgradpulse(1.0*ACQ_gzlvl3,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v12);
	ifrtEQ(v12,one,v13);
	 delay(tau_r-rof3);
	 zgradpulse(1.0*ACQ_gzlvl4,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	ifrtEQ(v12,two,v13);
	 delay(tau_r-rof3);
	 zgradpulse(1.0*ACQ_gzlvl1,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	ifrtEQ(v12,three,v13);
	 delay(tau_r-rof3);
	 zgradpulse(1.0*ACQ_gzlvl2,ACQ_gt1);
	 delay(ACQ_gstab);
	endif(v13);
	}
	else
	{
	ifzero(v12);
	 delay(ACQ_gstab);
	 zgradpulse(1.0*ACQ_gzlvl3,ACQ_gt1);
	 delay(tau_r-rof3);
	endif(v12);
	ifrtEQ(v12,one,v13);
	 delay(ACQ_gstab);
	 zgradpulse(1.0*ACQ_gzlvl4,ACQ_gt1);
	 delay(tau_r-rof3);
	endif(v13);
	ifrtEQ(v12,two,v13);
	 delay(ACQ_gstab);
	 zgradpulse(1.0*ACQ_gzlvl1,ACQ_gt1);
	 delay(tau_r-rof3);
	endif(v13);
	ifrtEQ(v12,three,v13);
	 delay(ACQ_gstab);
	 zgradpulse(1.0*ACQ_gzlvl2,ACQ_gt1);
	 delay(tau_r-rof3);
	endif(v13);
	}
	rcvron();
endloop(v11);
	sample((droppts1+kp_npoints+droppts2)/sw);
	recoff();	
	endacq();	
delay(0.05);
	if ( (lkgate_flg[0] == 'y') || (lkgate_flg[0] == 'k') )  lk_sample();
delay(0.05);

}



