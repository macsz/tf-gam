package pg.eti.biomed.leptonreader;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

public class FileReader {

    final static EventAggregator ea = new EventAggregator();

    public static void main(String[] args) {
        String path = "";
        try{
            path = args[0];
        }
        catch(Exception ex){
            path = "/home/macsz/Projects/deep_learning/desktop/evaluation/loud/LeptonData1479745019718.lep";
        }
        
//        Main previewWindow = new Main(ea);
        //Magifier m = new Magifier(ea);
        ManualVideoCompile manualVideoCompile = new ManualVideoCompile(ea);
//        previewWindow.setVisible(true);
        Show show = new Show();
        show.readLeptonFileWithBytesArray(path, 0, 1000000000);
//        File curDir = new File("/Users/ala/Movies/lepton_embc/training/");
//        File[] filesList = curDir.listFiles();
//        for(int i=0; i<filesList.length; i++){
//            
//            if(filesList[i].getName().endsWith(".lep") && i>19){
//                System.out.println(i);
//                System.out.println(filesList[i].getAbsolutePath());
//                show.readLeptonFileWithBytesArray(filesList[i].getAbsolutePath(), 0, 1000000000);
//            }
//        }
    }

    private static class Show {

        private int index;
        private int frameWidth;
        private int frameHeight;
        private int sizeOfFrame;
        private int firstFrame;
        private int totalNoOfFrames;
        private int noOfFrames;
        private BufferedImage[] images;
        private double[][] avgVector;
        private long[] dates;
        private double Ts;
        private double fs;
        private double samplingPeriod = 0;

        public Show() {
        }

        public void readLeptonFileWithBytesArray(String path, int firstFrame, int noOfFrames) {
            FileInputStream fis = null;
            try {
                BufferedImage img;
                File f = new File(path);
                long fileLength = f.length();
                fis = new FileInputStream(f);
                index = 0;
                frameWidth = 80;
                frameHeight = 60;
                //2Bytes per pixel 14bits resolution
                sizeOfFrame = frameWidth * frameHeight * 2;
                this.firstFrame = firstFrame;
                totalNoOfFrames = (int) (fileLength / sizeOfFrame);
                if (noOfFrames == 0 || noOfFrames > totalNoOfFrames) {
                    noOfFrames = totalNoOfFrames;

                }
                this.noOfFrames = noOfFrames;
                System.out.println("SIZE: " + sizeOfFrame + ", w=" + frameWidth + ", h=" + frameHeight + ", Total=" + totalNoOfFrames);
                images = new BufferedImage[noOfFrames];
                avgVector = new double[3][noOfFrames];
                dates = new long[noOfFrames];
                int[] pixels;
                //write and read image width and height
                byte[] b;
                //skip to firstFrame!!!
                if (firstFrame > 0) {
                    fis.skip(firstFrame * sizeOfFrame);
                }
                for (int l = firstFrame; l < (firstFrame + noOfFrames); l++) {
                    b = new byte[sizeOfFrame];
                    fis.read(b);

                    pixels = convertLeptonToInt(b, frameWidth, frameHeight);
                    //calculateScaleParams(1.025f,0.99f);
                    //calculateScaleParams2(0.1f, 0.01f);
                    calculateScaleParams2(0.2f, 0.02f);
                    pixels = convertLeptonToRGB(pixels, frameWidth, frameHeight);
                    //img = getImageFromArray2(pixels, frameWidth, frameHeight);
                    BufferedImage bufferedImage = new BufferedImage(frameWidth, frameHeight, BufferedImage.TYPE_INT_RGB);
                    int row = 0;

                    for (int i = 0; i < pixels.length; i++) {
                        try {
                            bufferedImage.setRGB(i % frameWidth, row, pixels[i]);

                        } catch (Exception e) {
                        }
                        if (i % frameWidth == 0) {
                            row++;
                        }
                    }

                    ea.triggerEvent(Event.PROCCESSED_IMAGE, bufferedImage, path);
                    dates[l - firstFrame] = 100;
                }
                //System.out.println("DATES: "+Arrays.toString(dates));
                samplingPeriod = 1 / 13.0f;
                Ts = samplingPeriod;
                fs = 1 / Ts;
                System.out.println("FS= " + fs + ", TS=" + Ts);
                fis.close();
                ea.triggerEvent(Event.FINISHED_READING, path, path);
            } catch (FileNotFoundException ex) {
                Logger.getLogger(FileReader.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(FileReader.class.getName()).log(Level.SEVERE, null, ex);
            } finally {
                try {
                    fis.close();
                } catch (IOException ex) {
                    Logger.getLogger(FileReader.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }

        int max = -Integer.MAX_VALUE;
        int min = Integer.MAX_VALUE;
        float a, b;

        public int[] convertLeptonToInt(byte[] data, int width, int height) {
            int size = width * height;
            int offset = size;
            int[] pixels = new int[size];

            int j = 0;
            for (int i = 0; i < size; i++) {
                j = i << 1;
                pixels[i] = ((data[j + 1] & 0xFF) << 8) | (data[j] & 0xFF);
            }

            for (int i = 0; i < pixels.length; i++) {
                if (pixels[i] < min) {
                    min = pixels[i];
                }
                if (pixels[i] > max) {
                    max = pixels[i];
                }
            }
            a = 255.0f / ((float) max - (float) min);
            b = -a * (float) min;
            return pixels;
        }

        public int[] histogram(int[] pixels) {
            if (max == -Integer.MAX_VALUE) {
                return null;
            }
            int[] histogram = new int[max - min];
            for (int i = 0; i < pixels.length; i++) {
                histogram[(pixels[i] - min)]++;
            }
            return histogram;
        }

        public void calculateScaleParams(float minScale, float maxScale) {
            int newMin = (int) (minScale * (float) min);
            int newMax = (int) (maxScale * (float) max);
            if (newMax <= newMin) {
//                System.out.println("WARNING NEW MAX < NEW MIN");
                return;
            }
            a = 255.0f / ((float) newMax - (float) newMin);
            b = -a * (float) newMin;
//            System.out.println("min=" + min + ", max=" + max + ", newMin=" + newMin + ", newMax=" + newMax + "a= " + a + ", b=" + b);
            min = newMin;
            max = newMax;
        }

        public void calculateScaleParams2(float minScale, float maxScale) {
            float newMin = 0;
            float newMax;
            float diff = max - min;

            newMin = min + (minScale * diff);
            newMax = max - (maxScale * diff);

            if (newMax <= newMin) {
//                System.out.println("WARNING NEW MAX < NEW MIN");
                return;
            }
            a = 255.0f / (newMax - newMin);
            b = -a * newMin;
//            System.out.println("min=" + min + ", max=" + max + ", newMin=" + newMin + ", newMax=" + newMax + "a= " + a + ", b=" + b);
            min = (int) newMin;
            max = (int) newMax;

        }

        public int[] convertLeptonToRGB(int[] pixels, int width, int height) {
            int size = width * height;
            if (max == -Integer.MAX_VALUE) {
                return null;
            }
            int vp;
            float v;
            for (int i = 0; i < pixels.length; i++) {
                vp = pixels[i];
                if (vp < min) {
                    vp = min;
                }
                if (vp > max) {
                    vp = max;
                }
                v = a * (float) vp + b;
                vp = ((int) v) & 0xFF;
                if (vp < 0) {
                    vp = 0;
                }
                if (vp > 255) {
                    vp = 255;
                }
                pixels[i] = (255 << 24) | (vp << 16) | (vp << 8) | vp;

            }
            return pixels;
        }
    }
}
