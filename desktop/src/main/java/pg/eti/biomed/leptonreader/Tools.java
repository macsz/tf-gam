/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package pg.eti.biomed.leptonreader;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;

/**
 *
 * @author ala
 */
public class Tools {

    public static BufferedImage DrawRectangle(BufferedImage sourceImage, int x, int y, int width, int height) {
        Graphics2D graph = sourceImage.createGraphics();
        graph.setColor(Color.GREEN);
        graph.draw(new Rectangle(x, y, width, height));
        graph.dispose();
        return sourceImage;
    }

    public static Color averageColor(BufferedImage bi, int x0, int y0, int width, int height) {
        long sumr = 0, sumg = 0, sumb = 0;
        for (int x = x0; x < x0 + width; x++) {
            for (int y = y0; y < y0 + height; y++) {
                Color pixel = new Color(bi.getRGB(x, y));
                sumr += pixel.getRed();
                sumg += pixel.getGreen();
                sumb += pixel.getBlue();
            }
        }
        int num = width * height;
        return new Color((int) sumr / num, (int) sumg / num, (int) sumb / num);
    }

    public static Color medianColor(BufferedImage bi, int x0, int y0, int width, int height) {
        ArrayList<Integer> reds = new ArrayList<>();
        ArrayList<Integer> greens = new ArrayList<>();
        ArrayList<Integer> blues = new ArrayList<>();
        for (int x = x0; x < x0 + width; x++) {
            for (int y = y0; y < y0 + height; y++) {
                Color pixel = new Color(bi.getRGB(x, y));
                reds.add(pixel.getRed());
                greens.add(pixel.getGreen());
                blues.add(pixel.getBlue());
            }
        }
        //sortowanie
        Collections.sort(reds);
        Collections.sort(greens);
        Collections.sort(blues);
        //mediany
        int r = reds.get((int) (reds.size() / 2.0));
        int g = greens.get((int) (greens.size() / 2.0));
        int b = blues.get((int) (blues.size() / 2.0));
        return new Color(r, g, b);
    }
}
