package main;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import algorithms.ItemPopularity;
import algorithms.MF_ALS;
import algorithms.MF_CD;
import algorithms.MF_fastALS;
import algorithms.MF_fastALS_plus;
import algorithms.MFbpr;
import data_structure.DenseMatrix;
import data_structure.Rating;
import data_structure.SparseMatrix;
import utils.Printer;

public class eals_plus extends main {
	public static void main(String[] argv)throws IOException{
		String dataset_name = "yelp";
		String method = "fastALS_plus";
		double w0 = 2000;
		boolean showProgress = true;
		boolean showLoss = true;
		int factors = 32;
		int maxIter = 50;
		double reg = 0.2;
		double alpha = 0.4;
		double lr = 0.01; 
		boolean adaptive = false;
		double parad = 0.2;
		SparseMatrix viewmatrix;
		boolean testmode = true ;
		double dstart = 0.5;
		double dend = 0.5;
		int dnum = 1;
		int showbound = 10 ;
		int showtime = 20;
		String viewfile = "C:\\Users\\\\thinkpad\\\\Documents\\\\GitHub\\\\dl\\\\deep\\\\sample\\\\cart_process";
		String datafile = "C:\\Users\\\\thinkpad\\\\Documents\\\\GitHub\\\\dl\\\\deep\\\\sample\\\\buy_process";
		// C:\Users\\thinkpad\\Documents\\GitHub\\dl\\deep\\sample\\buy_process
		 //String datafile ="data/yelp.rating";
		
		if (argv.length > 0) {
			dataset_name = argv[0];
			//method = argv[1];
			w0 = Double.parseDouble(argv[2]);
			showProgress = Boolean.parseBoolean(argv[3]);
			showLoss = Boolean.parseBoolean(argv[4]);
			factors = Integer.parseInt(argv[5]);
			maxIter = Integer.parseInt(argv[6]);
			reg = Double.parseDouble(argv[7]);
			if (argv.length > 8) alpha = Double.parseDouble(argv[8]);
			datafile = argv[9];
			viewfile = argv[10];
			testmode = Boolean.parseBoolean(argv[11]);
			dstart = Double.parseDouble(argv[12]);
			dend = Double.parseDouble(argv[13]);
			dnum = Integer.parseInt(argv[14]);
			if(argv.length > 15)
			{
				showbound = Integer.parseInt(argv[15]);
				showtime = Integer.parseInt(argv[16]);
			}
		}
		//ReadRatings_GlobalSplit("data/" + dataset_name + ".rating", 0.1);
		//ReadRatings_HoldOneOut("data/" + dataset_name + ".rating");
	    
		//filter
		ReadRatings_HoldOneOut(datafile);
		
		//view file to matrix 
		{
//			userCount = itemCount = 0;			
			// Step 1. Construct data structure for sorting.
			long startTime = System.currentTimeMillis();
			ArrayList<ArrayList<Rating>> user_ratings = new ArrayList<ArrayList<Rating>>();
			BufferedReader reader = new BufferedReader(
					new InputStreamReader(new FileInputStream(viewfile)));
			String line;
			while((line = reader.readLine()) != null) {
				Rating rating = new Rating(line);
				//if( rating.userId < 10000 && rating.itemId < 10000) {
				while (user_ratings.size()  < userCount) { // create a new user
					user_ratings.add(new ArrayList<Rating>());
				}
				if(rating.userId < userCount && rating.itemId < itemCount)
					user_ratings.get(rating.userId).add(rating);
//				userCount = Math.max(userCount, rating.userId);
//				itemCount = Math.max(itemCount, rating.itemId);
			}
			reader.close();
			//userCount ++;
			//itemCount ++;
			//assert userCount == user_ratings.size();
									
			// Step 3. Generated splitted matrices (implicit 0/1 settings). 
			System.out.printf("Generate view/buy matrics.");
			
			startTime = System.currentTimeMillis();
			viewmatrix = new SparseMatrix(userCount, itemCount);
			
			for (int u = 0; u < userCount; u ++) {
				//if(user_ratings.get(u).size()>0) {
				{
				ArrayList<Rating> ratings = user_ratings.get(u);
				for (int i = ratings.size() - 1; i >= 0; i --) {
					int userId = ratings.get(i).userId;
					int itemId = ratings.get(i).itemId;										
					viewmatrix.setValue(userId, itemId, 1);		
				}
				}
			}			
			System.out.printf("[%s]\n", Printer.printTime(
					System.currentTimeMillis() - startTime));			
			// Print some basic statistics of the dataset.
			System.out.println ("Data\t" + viewfile);
			System.out.printf("#Ratings\t %d (train)\n", 
					viewmatrix.itemCount());
		}		
		
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		evaluate_model(popularity, "Popularity");
		
		double init_mean = 0;
		double init_stdev = 0.01;
		
		if (testmode == true)
			for(parad = dstart ; parad <= dend; parad += (dend-dstart)/(dnum-1)){
				System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%f, w0=%.2f, alpha=%.2f,parad = %.2f\n",
						method, showProgress, factors, maxIter, reg, w0, alpha,parad);
				if (method.equalsIgnoreCase("fastals_plus")) {
					MF_fastALS_plus fals = new MF_fastALS_plus(trainMatrix, testRatings, topK, threadNum,
							factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss,
							viewmatrix,parad,showbound,showtime);
					evaluate_model(fals, "MF_fastALS_plus");
				}
				
				
			}
		else {
			System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%f, w0=%.2f, alpha=%.2f,parad = %.2f\n",
					method, showProgress, factors, maxIter, reg, w0, alpha,parad);
			if (method.equalsIgnoreCase("fastals_plus")) {
				MF_fastALS_plus fals = new MF_fastALS_plus(trainMatrix, testRatings, topK, threadNum,
						factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss,
						viewmatrix,parad,showbound,showtime);
				
				evaluate_model(fals, "MF_fastALS_plus");
			}
	    }	
	}
}