/**
 * Created by sunlu on 9/15/15.
 */


import meka.classifiers.multilabel.BCC;
import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.MULAN;


public class PerformMLC {

    static String n = "5";         // n-fold
    static String percent = "66.0"; // split percentage
    static String outputType = "2"; // 1, 2, 3, 4, 5, 6
    static String baseline = "Logistic"; // SMO, Logistic
    static String arfflist[] = { "scene", "emotions", "flags", "yeast",
            "birds", "genbase", "medical", "enron", "languagelog", "mediamill", "bibtex",
            "Corel5k" };

    static String[] gOptions = new String[12];  // general options
    static String[] options;                    // specified options

    public static void  main(String[] args) throws Exception {

        BR testClassifier0 = new BR();
        CC testClassifier1 = new CC();
//        BCC testClassifier2 = new BCC();
        BCCpro testClassifier2 = new BCCpro();  // BCC built based on normalized MI matirx
        PACC testClassifier3 = new PACC();
        MULAN testClassifier4 = new MULAN();      // MLkNN
        BRFS testClassifier5 = new BRFS();        // BR-FS
//        BRFSpro testClassifier5 = new BRFSpro();        // BR-FS (filter+Cfs)
//        BRFSpro2 testClassifier5 = new BRFSpro2();        // BR-FS (Cfs+Wrapper)
//        BRFS_wrapper testClassifier5 new BRFS_wrapper();  // BR-FS (Wrapper)
//        BRFS_testT testClassifier5 = new BRFS_testT();        // BR-FS
        CCFS testClassifier6 = new CCFS();        // CC-FS
        BCCFS testClassifier7 = new BCCFS();      // BCC-FS
        PACCFS testClassifier8 = new PACCFS();        // PACC-FS
        PACCFS_I testClassifier9 = new PACCFS_I();    // PACC-Filter
        PACCFS_II testClassifier10 = new PACCFS_II();    // PACC-Wrapper

        // **************************************************************
        // *************** Evaluate a single method *********************
        // **************************************************************
        String filename = "genbase";

        int i = 6;

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
                Evaluation.runExperiment(testClassifier2, options);
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
        }
        // **************************************************************
        // **************************************************************
        // **************************************************************



//		 //**************************************************************
//		 //************** Experiments on all methods ********************
//		 //**************************************************************
//		 for (int i = 0 ; i < 1; i ++) { // traverse all data sets
//
//			 setTestOptions(arfflist[i], "br", 1, 0);
//			 Evaluation.runExperiment(testClassifier0, options);
//
//			 setTestOptions(arfflist[i], "cc", 1, 0);
//			 Evaluation.runExperiment(testClassifier1, options);
//
//			 setTestOptions(arfflist[i], "bcc", 0, 0);
//			 Evaluation.runExperiment(testClassifier2, options);
//
//             setTestOptions(arfflist[i], "pacc", 0, 0);
//             Evaluation.runExperiment(testClassifier3, options);
//
//			 setTestOptions(arfflist[i], "mlknn", 0, 1);
//			 Evaluation.runExperiment(testClassifier4, options);
//
//			 setTestOptions(arfflist[i], "brfs", 0, 0);
//			 Evaluation.runExperiment(testClassifier5, options);
//
//			 setTestOptions(arfflist[i], "ccfs", 0, 0);
//			 Evaluation.runExperiment(testClassifier6, options);
//
//             setTestOptions(arfflist[i], "bccfs", 0, 0);
//             Evaluation.runExperiment(testClassifier8, options);
//
//			 setTestOptions(arfflist[i], "paccfs", 0, 0);
//			 Evaluation.runExperiment(testClassifier8, options);
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

//		// use n-fold cross validation // comment this snippet when using
//		// splitting
//		gOptions[2] = "-x";
//		gOptions[3] = n;

        // split train/test in percent%
        gOptions[2] = "-split-percentage";
        gOptions[3] = percent;

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
            options[12] = "MLCFS/" + baseline + "/" + arffname + "/" + method;
        }
        // *************** Meka & save file ********************

        // *************** Mulan & save file ********************
        if (sd == 0 && ml == 1) {

            options = new String[15];

            for (int i = 0; i < 11; i++) {
                options[i] = gOptions[i];
            }

            options[11] = "-f";
            options[12] = "MLCFS/" + baseline + "/" + arffname + "/" + method;
            options[13] = "-S";
            if (method == "clr")
                options[14] = "CLR";
            else
                options[14] = "MLkNN"; // default: MLkNN
        }
        // *************** Mulan & save file ********************

    }

}



