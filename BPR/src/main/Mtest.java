package main;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import utils.Printer;
import data_structure.DenseMatrix;

import utils.Printer;
import algorithms.MF_fastALS;
import algorithms.MFbpr;
import algorithms.MF_ALS;
import algorithms.MF_CD;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;
import algorithms.ItemPopularity;
import java.lang.*;

public class Mtest  {
	public static void main(String argv[]) {
		double u,vi,vj,k1,k2;
		u = 1;
		vi = 1;
		vj = -1;
		k2 = 0.01;
		k1 = 0.0002;
		for( int i = 1;i<100000;i++) {
			u = (1+k1)*u -k2*(vi-vj);
			vi = (1+k1)*vi - k2*u;
			vj = (1+k1)*vj + k2*u;
			System.out.printf("u = %f,vi = %f,vj =%f\n",u,vi,vj);
		}
			
	}
	
}