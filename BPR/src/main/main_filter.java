package main;

import java.io.IOException;

import data_structure.DenseMatrix;
import utils.Printer;
import algorithms.MF_fastALS;
import algorithms.MFbpr;
import algorithms.MF_ALS;
import algorithms.MF_CD;
import algorithms.ItemPopularity;

public class main_filter extends main {
	public static void main(String argv[]) throws IOException {
		String dataset_name = "yelp";
		String method = "bpr";
		double w0 = 1;
		boolean showProgress = false;
		boolean showLoss = false;
		int factors = 32;
		int maxIter = 4;
		double reg = 0.01;
		double alpha = 0.4;
		double lr = 0.01; 
		boolean adaptive = false;
		int userN = 10;
		int itemN = 10;
		String datafile = "C:\\Users\\\\thinkpad\\\\Documents\\\\GitHub\\\\dl\\\\deep\\\\sample\\\\buy_process";
		String viewfile = "C:\\Users\\\\thinkpad\\\\Documents\\\\GitHub\\\\dl\\\\deep\\\\sample\\\\collect_process";
		// C:\Users\\thinkpad\\Documents\\GitHub\\dl\\deep\\sample\\buy_process_
		// String datafile ="data/yelp.rating";
		
		if (argv.length > 0) {
			datafile = argv[0];
			viewfile = argv[1];
			userN = Integer.parseInt(argv[2]);
			itemN = Integer.parseInt(argv[3]);
			
		}
		//ReadRatings_GlobalSplit("data/" + dataset_name + ".rating", 0.1);
		//ReadRatings_HoldOneOut("data/" + dataset_name + ".rating");
		FilterRatingsWithThreshold(datafile,viewfile,userN,itemN);
		

	
	} // end main
}
