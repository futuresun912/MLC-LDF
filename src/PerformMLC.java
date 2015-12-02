/**
 * Created by sunlu on 9/15/15.
 */


import meka.classifiers.multilabel.*;


public class PerformMLC {

    static String n = "5";         // n-fold
    static String percent = "75.0"; // split percentage
    static String outputType = "2"; // 1, 2, 3, 4, 5, 6
    static String baseline = "Logistic"; // SMO, Logistic
    static String[] gOptions = new String[12];  // general options
    static String[] options;                    // specified options
    static String arfflist[] = {
            "scene",        // 0
            "emotions",     // 1
            "flags",        // 2
            "yeast",        // 3
            "birds",        // 4
            "genbase",      // 5
            "medical",      // 6
            "enron",        // 7
            "languagelog",  // 8

            "bibtex",       // 9
            "Corel5k",      // 10
            "mediamill",    // 11

//
//            "CAL500",       // 9
//            "rcv1subset1",   // 11
//            "rcv1subset3",   // 12
//            "delicious"      // 15

    };

    public static void  main(String[] args) throws Exception {

        BR testClassifier0 = new BR();
        CC testClassifier1 = new CC();
        BCC testClassifier2 = new BCC();
        BCCpro testClassifier21 = new BCCpro();  // BCC built on normalized MI matirx
        PCC testClassifierX = new PCC();
        PACC testClassifier3 = new PACC();
        MULAN testClassifier4 = new MULAN();      // MLkNN


        // **************************************************************
        // the following seven XXFS class is unavailable
        BRFS testClassifier5 = new BRFS();        // BR-FS
        CCFS testClassifier6 = new CCFS();        // CC-FS
        BCCFS testClassifier7 = new BCCFS();      // BCC-FS
        PACCFS testClassifier8 = new PACCFS();        // PACC-(Cfs+Wrapper)
        PACCFS_II testClassifier10 = new PACCFS_II();    // PACC-Wrapper
        PACCFS_G testClassifier11 = new PACCFS_G();    // PACC-IG
        PACCFS_III testClassifier12 = new PACCFS_III();   //PACC-(IG+Cfs)
        // **************************************************************


        // **************************************************************
        // **************************************************************
        PACCFS_IG_test testClassifier99 = new PACCFS_IG_test();   // PACC-IG
        PACCFS_I testClassifier9 = new PACCFS_I();                // PACC-CFS
        PACCFS_III_test testClassifier98 = new PACCFS_III_test(); // PACC-IG+Cfs

        CCLDF_IG testClassifier62 = new CCLDF_IG();               // CC-IG
        CCFS_I testClassifier61 = new CCFS_I();                   // CC-CFS
        CCLDF testClassifier63 = new CCLDF();                     // CC-IG+CFS

        BCCLDF_IG testClassifier72 = new BCCLDF_IG();             // BCC-IG
        BCCFS_I testClassifier71 = new BCCFS_I();                 // BCC-CFS
        BCCLDF testClassifier73 = new BCCLDF();                   // BCC-IG+CFS

        BRLDF_IG testClassifier52 = new BRLDF_IG();               // BR-IG
        BRFS_I testClassifier51 = new BRFS_I();                   // BR-CFS
        BRLDF testClassifier53 = new BRLDF();                     // BR-IG+CFS


        // **************************************************************
        // *************** Evaluate a single method *********************
        // **************************************************************

        String filename = arfflist[1];
//        String filename = "birds";

        int i = 2;

        switch (i) {
            case 0:
                setTestOptions(filename, "br", 1, 0);
                Evaluation.runExperiment(testClassifier0, options);
                break;
            case 1:
                setTestOptions(filename, "cc", 1, 0);
                Evaluation.runExperiment(testClassifier1, options);
                break;
            case 2:
                setTestOptions(filename, "bcc", 1, 0);
                Evaluation.runExperiment(testClassifier21, options);
                break;
            case 32:
                setTestOptions(filename, "pcc", 1, 0);
                Evaluation.runExperiment(testClassifierX, options);
                break;
            case 3:
                setTestOptions(filename, "pacc", 1, 0);
                Evaluation.runExperiment(testClassifier3, options);
                break;
            case 4:
                setTestOptions(filename, "mlknn", 1, 1);
                Evaluation.runExperiment(testClassifier4, options);
                break;
            case 5:
                setTestOptions(filename, "brfs", 1, 0);
                Evaluation.runExperiment(testClassifier5, options);
                break;
            case 6:
                setTestOptions(filename, "ccfs", 1, 0);
                Evaluation.runExperiment(testClassifier6, options);
                break;
            case 61:
                setTestOptions(filename, "ccfs-I", 1, 0);
                Evaluation.runExperiment(testClassifier61, options);
                break;
            case 7:
                setTestOptions(filename, "bccfs", 1, 0);
                Evaluation.runExperiment(testClassifier7, options);
                break;
            case 8:
                setTestOptions(filename, "paccfs", 1, 0);
                Evaluation.runExperiment(testClassifier8, options);
                break;
            case 9:
                setTestOptions(filename, "paccfs-I", 1, 0);
                Evaluation.runExperiment(testClassifier9, options);
                break;
            case 10:
                setTestOptions(filename, "paccfs-II", 1, 0);
                Evaluation.runExperiment(testClassifier10, options);
                break;
            case 11:
                setTestOptions(filename, "paccfs-G", 1, 0);
                Evaluation.runExperiment(testClassifier11, options);
                break;
            case 12:
                setTestOptions(filename, "paccfs-III", 1, 0);
                Evaluation.runExperiment(testClassifier12, options);
                break;
            case 99:
                setTestOptions(filename, "paccfs-IG-test", 1, 0);
                Evaluation.runExperiment(testClassifier99, options);
                break;
            case 98:
                setTestOptions(filename, "paccfs-III-test", 1, 0);
                Evaluation.runExperiment(testClassifier98, options);
                break;
            case 62:
                setTestOptions(filename, "ccldf_IG", 1, 0);
                Evaluation.runExperiment(testClassifier62, options);
                break;
            case 63:
                setTestOptions(filename, "ccldf", 1, 0);
                Evaluation.runExperiment(testClassifier63, options);
                break;
            case 72:
                setTestOptions(filename, "bccldf_IG", 1, 0);
                Evaluation.runExperiment(testClassifier72, options);
                break;
            case 71:
                setTestOptions(filename, "bccldf_CFS", 1, 0);
                Evaluation.runExperiment(testClassifier71, options);
                break;
            case 73:
                setTestOptions(filename, "bccldf", 1, 0);
                Evaluation.runExperiment(testClassifier73, options);
                break;
            case 52:
                setTestOptions(filename, "brldf_IG", 1, 0);
                Evaluation.runExperiment(testClassifier52, options);
                break;
            case 51:
                setTestOptions(filename, "brldf_CFS", 1, 0);
                Evaluation.runExperiment(testClassifier51, options);
                break;
            case 53:
                setTestOptions(filename, "brldf", 1, 0);
                Evaluation.runExperiment(testClassifier53, options);
                break;
        }
        // **************************************************************
        // **************************************************************
        // **************************************************************



//		 //**************************************************************
//		 //************** Experiments on all methods ********************
//		 //**************************************************************
//		 for (int i = 0 ; i < arfflist.length; i ++) { // traverse all data sets
//
////			 setTestOptions(arfflist[i], "br", 1, 0);
////			 EvaluationPro.runExperiment(testClassifier0, options);
////
//////			 setTestOptions(arfflist[i], "bcc", 1, 0);
//////			 EvaluationPro.runExperiment(testClassifier2, options);
////
////             setTestOptions(arfflist[i], "bccpro", 1, 0);
////             EvaluationPro.runExperiment(testClassifier21, options);
////
////             setTestOptions(arfflist[i], "pacc", 1, 0);
////             EvaluationPro.runExperiment(testClassifier3, options);
////
////			 setTestOptions(arfflist[i], "mlknn", 1, 1);
////			 EvaluationPro.runExperiment(testClassifier4, options);
////
////			 setTestOptions(arfflist[i], "brfs", 1, 0);
////			 EvaluationPro.runExperiment(testClassifier5, options);
////
////             setTestOptions(arfflist[i], "brfsI", 1, 0);
////             EvaluationPro.runExperiment(testClassifier51, options);
////
////             setTestOptions(arfflist[i], "bccfs", 1, 0);
////             EvaluationPro.runExperiment(testClassifier7, options);
////
////             setTestOptions(arfflist[i], "paccfs", 1, 0);
////             EvaluationPro.runExperiment(testClassifier8, options);
////
////             setTestOptions(arfflist[i], "bccfsI", 1, 0);
////             EvaluationPro.runExperiment(testClassifier71, options);
////
////             setTestOptions(arfflist[i], "paccfs-I", 1, 0);
////             EvaluationPro.runExperiment(testClassifier9, options);
//
////             setTestOptions(arfflist[i], "paccfs-II", 1, 0);
////             EvaluationPro.runExperiment(testClassifier10, options);
//
//
////             // CC seems need to be performed separated from others
////             setTestOptions(arfflist[i], "cc", 1, 0);
////             EvaluationPro.runExperiment(testClassifier1, options);
//
////             setTestOptions(arfflist[i], "ccfs", 1, 0);
////             EvaluationPro.runExperiment(testClassifier6, options);
////
////             setTestOptions(arfflist[i], "ccfsI", 1, 0);
////             EvaluationPro.runExperiment(testClassifier61, options);
//
//             System.out.println("*****************************************");
//             System.out.println("data-"+i+" starts!");
//             System.out.println("*****************************************\n");
//
//             setTestOptions(arfflist[i], "paccfs-IG-test", 1, 0);
//             EvaluationPro.runExperiment(testClassifier99, options);
//
////             setTestOptions(arfflist[i], "paccfs-I", 1, 0);
////             EvaluationPro.runExperiment(testClassifier9, options);
//
//             setTestOptions(arfflist[i], "paccfs-III-test", 1, 0);
//             EvaluationPro.runExperiment(testClassifier98, options);
//
////             setTestOptions(arfflist[i], "ccldf_IG", 1, 0);
////             EvaluationPro.runExperiment(testClassifier62, options);
////
////             setTestOptions(arfflist[i], "ccfs-I", 1, 0);
////             EvaluationPro.runExperiment(testClassifier61, options);
////
////             setTestOptions(arfflist[i], "ccldf", 1, 0);
////             EvaluationPro.runExperiment(testClassifier63, options);
////
//             setTestOptions(arfflist[i], "bccldf_IG", 1, 0);
//             EvaluationPro.runExperiment(testClassifier72, options);
//
////             setTestOptions(arfflist[i], "bccldf_CFS", 1, 0);
////             EvaluationPro.runExperiment(testClassifier71, options);
//
//             setTestOptions(arfflist[i], "bccldf", 1, 0);
//             EvaluationPro.runExperiment(testClassifier73, options);
//
//             setTestOptions(arfflist[i], "brldf_IG", 1, 0);
//             EvaluationPro.runExperiment(testClassifier52, options);
//
////             setTestOptions(arfflist[i], "brldf_CFS", 1, 0);
////             EvaluationPro.runExperiment(testClassifier51, options);
//
//             setTestOptions(arfflist[i], "brldf", 1, 0);
//             EvaluationPro.runExperiment(testClassifier53, options);
//
//             System.out.println("*****************************************");
//             System.out.println("data-"+i+" is finished!");
//             System.out.println("*****************************************\n");
//
//		 }
//	   //**************************************************************
//	   //**************************************************************
//	   //**************************************************************

    }

    public static void setTestOptions(String arffname, String method, int sd,
                                      int ml) {       // sd: save(0) or debug(1);
        // ml: meka(0) or mulan(1)

        // *************** general options ********************

        // select a data set
        gOptions[0] = "-t";
        gOptions[1] = "/home/sunlu/workspace/data/" + arffname + ".arff";

		// use n-fold cross validation // comment this snippet when using
		gOptions[2] = "-x";
		gOptions[3] = n;

//        // split train/test in percent%
//        gOptions[2] = "-split-percentage";
//        gOptions[3] = percent;

        // unknown
        gOptions[4] = "-s";
        gOptions[5] = "1";
        gOptions[6] = "-R";

        // output type
        gOptions[7] = "-verbosity";
        gOptions[8] = outputType;

        // choose the baseline classifier
        gOptions[9] = "-W";
        gOptions[10] = "weka.classifiers.functions." + baseline;
//		gOptions[10] = "weka.classifiers.bayes.NaiveBayes";

        // output debug information
        gOptions[11] = "-output-debug-info";
//        gOptions[11] = "";

        // *************** general options ********************

        // *************** Meka & debug ********************
        if (sd == 1 && ml == 0) {

            options = new String[12];

            for (int i = 0; i < 12; i++) {
                options[i] = gOptions[i];
            }

        }

        // *************** Mulan & debug ********************
        if (sd == 1 && ml == 1) {

            options = new String[14];

            for (int i = 0; i < 12; i++) {
                options[i] = gOptions[i];
            }

            options[12] = "-S";
            if (method == "clr")
                options[13] = "CLR";
            else
                options[13] = "MLkNN"; // default: MLkNN
        }
        // *************** Mulan & debug ********************

        // *************** Meka & save file ********************
        if (sd == 0 && ml == 0) {

            options = new String[13];
            // weka.classifiers.bayes.NaiveBayes
            for (int i = 0; i < 11; i++) {
                options[i] = gOptions[i];
            }

            options[11] = "-f";
            options[12] = "/home/sunlu/experiments/" + arffname + "/" + method;
        }
        // *************** Meka & save file ********************

        // *************** Mulan & save file ********************
        if (sd == 0 && ml == 1) {

            options = new String[15];

            for (int i = 0; i < 11; i++) {
                options[i] = gOptions[i];
            }

            options[11] = "-f";
            options[12] = "experiments/" + arffname + "/" + method;
            options[13] = "-S";
            if (method == "clr")
                options[14] = "CLR";
            else
                options[14] = "MLkNN"; // default: MLkNN
        }
        // *************** Mulan & save file ********************

    }

}



