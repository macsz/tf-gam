package pg.eti.biomed.leptonreader;

import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.util.concurrent.TimeUnit;

import com.xuggle.mediatool.IMediaWriter;
import com.xuggle.mediatool.ToolFactory;
import com.xuggle.xuggler.ICodec;
import java.awt.Point;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

public class ManualVideoCompile implements EventListener {

    private static Dimension screenBounds;
    public static int indexVideo = 1;

    private static final String OUTPUT_FILE = "today.mp4";
    private final EventAggregator ea;
    private ArrayList<BufferedImage> images;
    private boolean firstFrame = true;
    int number = 0;
    static int x_pos, y_pos = 0;
    static int x_pos2, y_pos2 = 0;
    static String name = "";

    public ManualVideoCompile(EventAggregator ea) {
        this.ea = ea;
        ea.addEventListener(Event.PROCCESSED_IMAGE, this);
        ea.addEventListener(Event.FINISHED_READING, this);
        ea.addEventListener(Event.NEW_POSITION, this);
        ea.addEventListener(Event.NEW_POSITION2, this);
        images = new ArrayList<>();
    }

    private void createVideo() {
//        final IMediaWriter writer = ToolFactory.makeWriter(OUTPUT_FILE);
//        screenBounds = Toolkit.getDefaultToolkit().getScreenSize();
//        writer.addVideoStream(0, 0, ICodec.ID.CODEC_ID_MPEG4,
//                screenBounds.width / 2, screenBounds.height / 2);

        int time = 0, index = 0;
        for (BufferedImage bgrScreen : images) {
//            bgrScreen = convertToType(bgrScreen, BufferedImage.TYPE_3BYTE_BGR);

//            writer.encodeVideo(0, bgrScreen, time,
//                    TimeUnit.MILLISECONDS);
//            time += 30;
        }
        System.out.println("Finished");
//        writer.close();
    }

    public static void saveBuffImgToFile(BufferedImage img, String name) {
        File outputfile = new File(name + ".jpg");
        try {
            ImageIO.write(img, "jpg", outputfile);
        } catch (IOException ex) {
            Logger.getLogger(ManualVideoCompile.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static BufferedImage convertToType(BufferedImage sourceImage,
            int targetType) {

        BufferedImage image;

        // if the source image is already the target type, return the source
        // image
        if (sourceImage.getType() == targetType) {
            image = sourceImage;
        } // otherwise create a new image of the target type and draw the new
        // image
        else {
            image = new BufferedImage(sourceImage.getWidth(),
                    sourceImage.getHeight(), targetType);
            image.getGraphics().drawImage(sourceImage, 0, 0, null);
        }

        return image;

    }

    @Override
    public void onEventOccurred(Event event, Object parameter, Object parameter2) {
        if (event == Event.PROCCESSED_IMAGE) {
            number++;
            String n = String.format("%1$05d", number);
            saveBuffImgToFile((BufferedImage) parameter,
                    "/home/macsz/Projects/deep_learning/frames/frame_"+n);
            images.add((BufferedImage) parameter);
            name = (String) parameter2;
        } else if (event == Event.FINISHED_READING) {
            System.out.println("Finished saving frames!");
//            createVideo();
        } else if (event == Event.NEW_POSITION) {
            images.clear();
            x_pos = (Integer) parameter / 4;
            y_pos = (Integer) parameter2 / 4;
            System.out.println("x i y " + x_pos + " " + y_pos);
        } else if (event == Event.NEW_POSITION2) {

            x_pos2 = (Integer) parameter / 4;
            y_pos2 = (Integer) parameter2 / 4;
            System.out.println("x i y " + x_pos2 + " " + y_pos2);
        }
    }

    private static int getX() {
        return x_pos;
    }

    private static int getY() {
        return y_pos;
    }

    private static int getX2() {
        return x_pos2;
    }

    private static int getY2() {
        return y_pos2;
    }

    private static String getName() {
        String[] bits = name.split("/");
        return bits[bits.length - 1];
    }
}
