package com.tenbitmelon.machinelearningplayer.util;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.TextComponent;
import net.kyori.adventure.text.event.ClickEvent;
import net.kyori.adventure.text.serializer.plain.PlainTextComponentSerializer;
import org.apache.commons.lang3.tuple.MutableTriple;
import org.bytedeco.pytorch.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class Utils {

    private static final HashMap<Character, Integer> characterWidths = new HashMap<>() {{
        put(' ', 5);
        put('!', 2);
        put('"', 5);
        put('#', 6);
        put('$', 6);
        put('%', 6);
        put('&', 6);
        put('\'', 3);
        put('(', 5);
        put(')', 5);
        put('*', 5);
        put('+', 6);
        put(',', 2);
        put('-', 6);
        put('.', 2);
        put('/', 6);
        put('0', 6);
        put('1', 6);
        put('2', 6);
        put('3', 6);
        put('4', 6);
        put('5', 6);
        put('6', 6);
        put('7', 6);
        put('8', 6);
        put('9', 6);
        put(':', 2);
        put(';', 2);
        put('<', 5);
        put('=', 6);
        put('>', 5);
        put('?', 6);
        put('@', 7);
        put('A', 6);
        put('B', 6);
        put('C', 6);
        put('D', 6);
        put('E', 6);
        put('F', 6);
        put('G', 6);
        put('H', 6);
        put('I', 4);
        put('J', 6);
        put('K', 6);
        put('L', 6);
        put('M', 6);
        put('N', 6);
        put('O', 6);
        put('P', 6);
        put('Q', 6);
        put('R', 6);
        put('S', 6);
        put('T', 6);
        put('U', 6);
        put('V', 6);
        put('W', 6);
        put('X', 6);
        put('Y', 6);
        put('Z', 6);
        put('[', 4);
        put('\\', 6);
        put(']', 4);
        put('^', 6);
        put('_', 6);
        put('`', 3);
        put('a', 6);
        put('b', 6);
        put('c', 6);
        put('d', 6);
        put('e', 6);
        put('f', 5);
        put('g', 6);
        put('h', 6);
        put('i', 2);
        put('j', 6);
        put('k', 5);
        put('l', 3);
        put('m', 6);
        put('n', 6);
        put('o', 6);
        put('p', 6);
        put('q', 6);
        put('r', 6);
        put('s', 6);
        put('t', 4);
        put('u', 6);
        put('v', 6);
        put('w', 6);
        put('x', 6);
        put('y', 6);
        put('z', 6);
        put('{', 5);
        put('|', 2);
        put('}', 5);
        put('~', 7);
        put('¡', 2);
        put('¢', 4);
        put('£', 6);
        put('¤', 4);
        put('¥', 4);
        put('¦', 1);
        put('§', -1);
        put('¨', 3);
        put('©', 5);
        put('ª', 6);
        put('«', 6);
        put('¬', 6);
        put('®', 7);
        put('¯', 4);
        put('°', 7);
        put('±', 7);
        put('²', 6);
        put('³', 3);
        put('´', 2);
        put('µ', 3);
        put('¶', 4);
        put('·', 6);
        put('¸', 2);
        put('¹', 2);
        put('º', 6);
        put('»', 6);
        put('¼', 6);
        put('½', 6);
        put('¾', 4);
        put('¿', 6);
        put('À', 6);
        put('Á', 6);
        put('Â', 6);
        put('Ã', 4);
        put('Ä', 6);
        put('Å', 6);
        put('Æ', 6);
        put('Ç', 6);
        put('È', 6);
        put('É', 6);
        put('Ê', 6);
        put('Ë', 6);
        put('Ì', 3);
        put('Í', 4);
        put('Î', 3);
        put('Ï', 3);
        put('Ð', 4);
        put('Ñ', 6);
        put('Ò', 4);
        put('Ó', 6);
        put('Ô', 6);
        put('Õ', 6);
        put('Ö', 6);
        put('×', 4);
        put('Ø', 6);
        put('Ù', 4);
        put('Ú', 6);
        put('Û', 4);
        put('Ü', 6);
        put('Ý', 4);
        put('Þ', 4);
        put('ß', 6);
        put('à', 6);
        put('á', 6);
        put('â', 6);
        put('ã', 6);
        put('ä', 6);
        put('å', 6);
        put('æ', 6);
        put('ç', 6);
        put('è', 6);
        put('é', 6);
        put('ê', 6);
        put('ë', 6);
        put('ì', 3);
        put('í', 3);
        put('î', 6);
        put('ï', 4);
        put('ð', 4);
        put('ñ', 6);
        put('ò', 6);
        put('ó', 6);
        put('ô', 6);
        put('õ', 6);
        put('ö', 6);
        put('÷', 7);
        put('ø', 6);
        put('ù', 6);
        put('ú', 6);
        put('û', 6);
        put('ü', 6);
        put('ý', 4);
        put('þ', 3);
        put('←', 8);
        put('→', 8);
        put('↞', 5);
        put('↠', 5);
    }};

    public static int characterWidth(char c) {
        return characterWidths.getOrDefault(c, 0);
    }

    public static int stringWidth(String string) {
        return string.chars().map(i -> characterWidth((char) i)).sum();
    }

    public static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    public static double roundRotation(double rotation, int steps) {
        return Math.round(rotation / (360f / steps)) * (360f / steps);
    }

    public static List<MutableTriple<Integer, Integer, ClickEvent>> findClickEvents(Component component) {
        if (component == null) return null;

        List<MutableTriple<Integer, Integer, ClickEvent>> result = new ArrayList<>();
        findClickEventsRecursive(component, 0, result);
        return result;
    }

    private static int findClickEventsRecursive(Component component, int currentPosition, List<MutableTriple<Integer, Integer, ClickEvent>> clickEvents) {
        if (component == null) return 0;

        // Calculate the length of the component in characters
        int componentLength = 0;

        // If this component has a click event, record it
        if (component instanceof TextComponent textComponent && textComponent.clickEvent() != null) {
            componentLength = stringWidth(textComponent.content());
            clickEvents.add(MutableTriple.of(
                currentPosition,
                currentPosition + componentLength,
                component.clickEvent()
            ));
        }

        // Process all child components, tracking position as we go
        int childPosition = currentPosition + componentLength;
        for (Component child : component.children()) {
            int childLength = findClickEventsRecursive(child, childPosition, clickEvents);
            childPosition += childLength;
        }

        String componentText = PlainTextComponentSerializer.plainText().serialize(component);
        return stringWidth(componentText);
    }


    public static String tensorString(Tensor tensor) {
        StringBuilder sb = new StringBuilder();
        sb.append(Arrays.toString(tensor.shape()));
        sb.append(": ");
        formatTensor(tensor, sb, 0);
        return sb.toString();
    }

    private static void formatTensor(Tensor tensor, StringBuilder sb, int indent) {
        long[] shape = tensor.shape();

        if (shape.length == 0) {
            sb.append(tensor.item().toFloat());
            return;
        }

        if (shape.length == 1) {
            sb.append(" ".repeat(indent));
            sb.append("[");
            for (int i = 0; i < shape[0]; i++) {
                if (i > 0) sb.append(", ");
                sb.append(tensor.get(i).item().toFloat());
            }
            sb.append("]");
            return;
        }

        String indentStr = " ".repeat(indent);
        sb.append(indentStr).append("[");
        for (int i = 0; i < shape[0]; i++) {
            sb.append(indentStr).append("  ");
            formatTensor(tensor.get(i), sb, indent + 2);
            if (i < shape[0] - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append(indentStr).append("]");
    }

    // def szudzik_unpairing(index: Index0):
    // shell = floor(sqrt(index))
    //     if index - shell ^ 2 < shell:
    //     return [shell, index - shell ^ 2]
    //     else:
    //     return [index - shell ^ 2 - shell, shell]

    public static int[] szudzikUnpairing(int index) {
        int shell = (int) Math.floor(Math.sqrt(index));
        if (index - shell * shell < shell) {
            return new int[]{shell, index - shell * shell};
        } else {
            return new int[]{index - shell * shell - shell, shell};
        }
    }
}

