package main;

import java.io.IOException;

import data_structure.DenseMatrix;
import utils.Printer;
import algorithms.MF_fastALS;
import algorithms.MFbpr;
import algorithms.MFbpr2;
import algorithms.MF_ALS;
import algorithms.MF_CD;
import algorithms.ItemPopularity;

public class main_bpr extends main {
	public static void main(String argv[]) throws IOException {
		String dataset_name = "yelp";
		String method = "bpr";
		double w0 = 2000;
		boolean showProgress = true;
		boolean showLoss = false;
		int factors = 32;
		int maxIter = 10;
		double reg = 0;
		double alpha = 0.4;
		double lr = 0.01; 
		boolean adaptive = false;
		int paraK = 1;
		String datafile = "C:\\Users\\\\thinkpad\\\\Documents\\\\GitHub\\\\dl\\\\deep\\\\sample\\\\buy";
		int showbound = 400;
		int showcount = 10;
		// C:\Users\\thinkpad\\Documents\\GitHub\\dl\\deep\\sample\\buy_process_
		// String datafile ="data/yelp.rating";
		
		if (argv.length > 0) {
			//dataset_name = argv[0];
			//method = argv[1];
			w0 = Double.parseDouble(argv[2]);
			showProgress = Boolean.parseBoolean(argv[3]);
			showLoss = Boolean.parseBoolean(argv[4]);
			factors = Integer.parseInt(argv[5]);
			maxIter = Integer.parseInt(argv[6]);
			reg = Double.parseDouble(argv[7]);
			if (argv.length > 8) alpha = Double.parseDouble(argv[8]);
			datafile = argv[9];
			showbound = Integer.parseInt(argv[10]);
			showcount = Integer.parseInt(argv[11]);
			if (argv.length>12) paraK = Integer.parseInt(argv[12]);
		}
		//ReadRatings_GlobalSplit("data/" + dataset_name + ".rating", 0.1);
		//ReadRatings_HoldOneOut("data/" + dataset_name + ".rating");
		ReadRatings_HoldOneOut(datafile);
		
		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%.6f, w0=%.6f, alpha=%.4f,paraK=%d\n",
				method, showProgress, factors, maxIter, reg, w0, alpha, paraK);
		System.out.println("====================================================");
		
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		evaluate_model(popularity, "Popularity");
		
		double init_mean = 0;
		double init_stdev = 0.01;
		

		
		if (method.equalsIgnoreCase("bpr")) {
//			MFbpr bpr = new MFbpr(trainMatrix, testRatings, topK, threadNum,
//					factors, maxIter, w0, adaptive, reg, init_mean, init_stdev, showProgress);
			MFbpr2 bpr = new MFbpr2(trainMatrix, testRatings, topK, threadNum, 
					factors, maxIter, w0, false, reg, init_mean, init_stdev, showProgress,showbound,showcount,paraK);
			evaluate_model(bpr, "MFbpr2");
		}
		
//		if (method.equalsIgnoreCase("bpr")) {
////			MFbpr bpr = new MFbpr(trainMatrix, testRatings, topK, threadNum,
////					factors, maxIter, w0, adaptive, reg, init_mean, init_stdev, showProgress);
//			MFbpr bpr = new MFbpr(trainMatrix, testRatings, topK, threadNum, 
//					32, maxIter, 0.64, false, reg,0,0.01, true);
//			evaluate_model(bpr, "MFbpr");
//		}
		
		if (method.equalsIgnoreCase("all")) {
			DenseMatrix U = new DenseMatrix(userCount, factors);
			DenseMatrix V = new DenseMatrix(itemCount, factors);
			U.init(init_mean, init_stdev);
			V.init(init_mean, init_stdev);
			
			MF_fastALS fals = new MF_fastALS(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss);
			fals.setUV(U, V);
			evaluate_model(fals, "MF_fastALS");
			
			MF_ALS als = new MF_ALS(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss);
			als.setUV(U, V);
			evaluate_model(als, "MF_ALS");
			
			MF_CD cd = new MF_CD(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss);
			cd.setUV(U, V);
			evaluate_model(cd, "MF_CD");
		}
	
	} // end main
}
