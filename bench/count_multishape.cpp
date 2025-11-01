#include "surface_generator.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

static int fractionToCoord(int max_value, double fraction) {
    if (fraction < 0.0) fraction = 0.0;
    else if (fraction > 1.0) fraction = 1.0;
    int value = static_cast<int>(std::round(max_value * fraction));
    if (value < 0) value = 0;
    else if (value > max_value) value = max_value;
    return value;
}

static SurfaceIntervals buildCompositeRectangles(int width, int height) {
    SurfaceIntervals rects = generateRectangle(
        width, height,
        fractionToCoord(width, 0.05), fractionToCoord(width, 0.35),
        fractionToCoord(height, 0.10), fractionToCoord(height, 0.40));
    rects = unionSurfaces(rects, generateRectangle(
        width, height,
        fractionToCoord(width, 0.30), fractionToCoord(width, 0.55),
        fractionToCoord(height, 0.25), fractionToCoord(height, 0.55)));
    rects = unionSurfaces(rects, generateRectangle(
        width, height,
        fractionToCoord(width, 0.60), fractionToCoord(width, 0.90),
        fractionToCoord(height, 0.20), fractionToCoord(height, 0.50)));
    rects = unionSurfaces(rects, generateRectangle(
        width, height,
        fractionToCoord(width, 0.15), fractionToCoord(width, 0.40),
        fractionToCoord(height, 0.55), fractionToCoord(height, 0.85)));
    rects = unionSurfaces(rects, generateRectangle(
        width, height,
        fractionToCoord(width, 0.45), fractionToCoord(width, 0.75),
        fractionToCoord(height, 0.05), fractionToCoord(height, 0.25)));
    return rects;
}

static SurfaceIntervals buildCompositeCircles(int width, int height,
                                        double dx = 0.0, double dy = 0.0, double dr = 0.0) {
    auto makeCircle = [&](double cx, double cy, double radius_fraction) {
        int center_x = fractionToCoord(width, cx + dx);
        int center_y = fractionToCoord(height, cy + dy);
        int radius = std::max(1, fractionToCoord(width, radius_fraction + dr));
        return generateCircle(width, height, center_x, center_y, radius);
    };

    SurfaceIntervals circles = makeCircle(0.20, 0.30, 0.12);
    circles = unionSurfaces(circles, makeCircle(0.50, 0.20, 0.10));
    circles = unionSurfaces(circles, makeCircle(0.75, 0.60, 0.14));
    circles = unionSurfaces(circles, makeCircle(0.35, 0.75, 0.11));
    circles = unionSurfaces(circles, makeCircle(0.62, 0.48, 0.09));
    return circles;
}

int main() {
    const int rows2D = 1024;
    const int intervalsPerRow2D = 256;
    const int width = intervalsPerRow2D * 4;
    const int height = rows2D;
    const int multi_pairs = 8;
    long long totalA = 0, totalB = 0;
    for (int i = 0; i < multi_pairs; ++i) {
        double dx = (i % 4) * 0.03;
        double dy = (i / 4) * 0.02;
        double dr = (i % 3 == 0) ? -0.01 : 0.0;
        SurfaceIntervals rects = buildCompositeRectangles(width, height);
        SurfaceIntervals circs = buildCompositeCircles(width, height, dx, dy, dr);
        totalA += rects.intervalCount();
        totalB += circs.intervalCount();
        std::printf("pair %d: A=%d, B=%d\n", i, rects.intervalCount(), circs.intervalCount());
    }
    std::printf("A_total=%lld B_total=%lld per_pair_avg=%lld\n", totalA, totalB, (totalA+totalB)/multi_pairs);
    return 0;
}

