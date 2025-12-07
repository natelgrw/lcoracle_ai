import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface LabelData {
  main: string;
  sub: string;
  color: string;
  bgFill: string;
  stroke: string;
  accent: string;
}

const ChromatogramAnimation: React.FC = () => {
  const [visiblePeakCount, setVisiblePeakCount] = useState(4);
  const [scaleFactor, setScaleFactor] = useState(1);
  const [isTablet, setIsTablet] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [labelScale, setLabelScale] = useState(1);
  const [windowWidth, setWindowWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1024);

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      setWindowWidth(width);
      if (width < 640) {
        setVisiblePeakCount(1);
        setScaleFactor(2.5);
        setIsTablet(false);
        setIsMobile(true);
      } else if (width < 1024) {
        setVisiblePeakCount(3);
        setScaleFactor(1.7);
        setIsTablet(true);
        setIsMobile(false);
      } else {
        setVisiblePeakCount(4);
        setScaleFactor(1);
        setIsTablet(false);
        setIsMobile(false);
      }

      // Calculate label scale to maintain constant physical size
      // SVG viewBox width is 1000. Rendered width is min(windowWidth, 1024).
      // We scale up the label elements when the SVG is scaled down.
      const renderedWidth = Math.min(width, 1024);
      setLabelScale(1000 / renderedWidth);
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // peak heights scaled by device
  const allPeaks = [
    { id: 1, height: 180, width: 30, delay: 0 },
    { id: 2, height: 140, width: 25, delay: 2 },
    { id: 3, height: 260, width: 35, delay: 4 },
    { id: 4, height: 110, width: 20, delay: 1 },
  ];

  const peaks = allPeaks.slice(0, visiblePeakCount).map((p, i) => {
    const step = 1000 / (visiblePeakCount + 1);
    // Dynamic height calculation for mobile to maintain 30px gap
    // Formula derived from: PeakHeight = 400 - (60000 / Width)
    // This ensures the label top is ~50px above SVG top (30px below button).
    // For width <= 640px, we apply a linear boost: +50px height for every 30px width decrease.
    // Base boost at 640px is 40px.
    const boost = windowWidth <= 640 ? 40 + ((50 / 30) * (640 - windowWidth)) : 0;
    const mobileHeight = 400 - (60000 / windowWidth) + boost;

    return {
      ...p,
      x: Math.round(step * (i + 1)),
      height: isMobile ? mobileHeight : p.height * scaleFactor * (isTablet ? 0.6 : 1),
      width: p.width * scaleFactor
    };
  });

  // possible labels for peaks
  const possibleLabels = [
    { main: "Peak A", sub: "RT 4.2m" },
    { main: "Impurity", sub: "m/z 145.2" },
    { main: "Product", sub: "99.8%" },
    { main: "Isomer 1", sub: "m/z 302.1" },
    { main: "Fragment 1", sub: "m/z 77.0" },
    { main: "Unknown", sub: "No Match" },
    { main: "Adduct 1", sub: "[M+H]+" },
    { main: "Solvent 1", sub: "RT 2.1m" },
    { main: "Dimer", sub: "2M+Na" },
    { main: "Metabolite", sub: "M1" },
    { main: "Caffeine", sub: "m/z 195.1" },
    { main: "Benzene", sub: "RT 3.5m" },
    { main: "Peak B", sub: "m/z 124.8" },
    { main: "Peak C", sub: "No Match" },
    { main: "Peak D", sub: "m/z 265.2" },
    { main: "Peak E", sub: "44.8%" },
    { main: "Peak F", sub: "m/z 419.4" },
    { main: "Peak G", sub: "m/z 496.5" },
    { main: "Peak H", sub: "RT 7.8m" },
    { main: "Peak I", sub: "m/z 650.7" },
    { main: "Peak J", sub: "m/z 727.8" },
    { main: "Peak K", sub: "RT 17.8m" },
    { main: "Peak L", sub: "m/z 882.0" },
    { main: "Peak M", sub: "No Match" },
    { main: "Peak N", sub: "RT 3.1m" },
    { main: "Peak O", sub: "m/z 1113.3" },
    { main: "Peak P", sub: "m/z 1190.4" },
    { main: "Peak Q", sub: "m/z 1267.5" },
    { main: "Peak R", sub: "m/z 1344.6" },
    { main: "Peak S", sub: "m/z 1421.7" },
    { main: "Peak T", sub: "77.8%" },
    { main: "Hexane", sub: "88.2%" },
    { main: "Solvent 2", sub: "No Match" },
    { main: "Fragment 2", sub: "66.9%" },
    { main: "Isomer 2", sub: "m/z 301.1" },
    { main: "Adduct 3", sub: "[M+K]+" },
    { main: "Adduct 2", sub: "[M+Na]+" },
  ];

  // colors for peak labels
  const colors = [
    { text: "#2563eb", bg: "#eff6ff", stroke: "#60a5fa", accent: "#3b82f6" }, // Blue
    { text: "#9333ea", bg: "#faf5ff", stroke: "#c084fc", accent: "#a855f7" }, // Purple
    { text: "#16a34a", bg: "#f0fdf4", stroke: "#4ade80", accent: "#22c55e" }, // Green
    { text: "#dc2626", bg: "#fef2f2", stroke: "#f87171", accent: "#ef4444" }, // Red
    { text: "#ea580c", bg: "#fff7ed", stroke: "#fb923c", accent: "#f97316" }, // Orange
  ];

  const getRandomLabel = (): LabelData => {
    const content = possibleLabels[Math.floor(Math.random() * possibleLabels.length)];
    const theme = colors[Math.floor(Math.random() * colors.length)];
    return { ...content, color: theme.text, bgFill: theme.bg, stroke: theme.stroke, accent: theme.accent };
  };

  const [labels, setLabels] = useState<{ [key: number]: LabelData }>({});

  useEffect(() => {
    setLabels(prev => {
      const next = { ...prev };
      allPeaks.forEach(p => {
        if (!next[p.id]) next[p.id] = getRandomLabel();
      });
      return next;
    });
  }, []);

  useEffect(() => {
    const cycleDuration = 12000;
    const timers: ReturnType<typeof setTimeout>[] = [];

    allPeaks.forEach(peak => {
      const initialDelay = (peak.delay * 1000) + 10000;
      const timer = setTimeout(() => {
        setLabels(prev => ({ ...prev, [peak.id]: getRandomLabel() }));
        const interval = setInterval(() => {
          setLabels(prev => ({ ...prev, [peak.id]: getRandomLabel() }));
        }, cycleDuration);
        timers.push(interval);
      }, initialDelay);
      timers.push(timer);
    });

    return () => timers.forEach(t => clearTimeout(t as any));
  }, []);

  const yBase = 350 + (scaleFactor > 1 ? 50 : 0);
  const viewBoxHeight = 400 + (scaleFactor > 1 ? 100 : 0);

  const getPeakPath = (cx: number, height: number, width: number) => {
    const w = width * 2;
    const h = height;
    return `M ${cx - w} ${yBase} C ${cx - w / 2} ${yBase}, ${cx - w / 2} ${yBase - h}, ${cx} ${yBase - h} C ${cx + w / 2} ${yBase - h}, ${cx + w / 2} ${yBase}, ${cx + w} ${yBase}`;
  };

  return (
    <div className="w-full h-[400px] relative flex items-end justify-center">
      <svg className="w-full h-full max-w-5xl overflow-visible" viewBox={`0 0 1000 ${viewBoxHeight}`} preserveAspectRatio="xMidYMax meet">
        <defs>
          <linearGradient id="chromaGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#3b82f6" />
            <stop offset="50%" stopColor="#a855f7" />
            <stop offset="100%" stopColor="#ec4899" />
          </linearGradient>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <motion.path
          d={`M 0 ${yBase} L 1000 ${yBase}`}
          stroke="url(#chromaGradient)"
          strokeWidth="2"
          strokeOpacity="0.3"
          fill="none"
        />

        {peaks.map((peak) => (
          <PeakComponent
            key={peak.id}
            peak={peak}
            labelData={labels[peak.id] || getRandomLabel()}
            getPeakPath={getPeakPath}
            yBase={yBase}
            labelScale={labelScale}
            windowWidth={windowWidth}
          />
        ))}
      </svg>
    </div>
  );
};

const PeakComponent: React.FC<{
  peak: { id: number, x: number, height: number, width: number, delay: number };
  labelData: LabelData;
  getPeakPath: (x: number, h: number, w: number) => string;
  yBase: number;
  labelScale: number;
  windowWidth: number;
}> = ({ peak, labelData, getPeakPath, yBase, labelScale, windowWidth }) => {

  const path = getPeakPath(peak.x, peak.height, peak.width);
  // Base dimensions (desktop)
  const baseWidth = 140;
  const baseHeight = 44;
  const baseFontSizeMain = 14;
  const baseFontSizeSub = 11;
  const baseGap = 65;

  // Increase connector gap slightly for very small screens (<= 550px)
  const connectorOffset = windowWidth <= 550 ? 12 : 10;

  // Scaled dimensions
  const cardWidth = baseWidth * labelScale;
  const cardHeight = baseHeight * labelScale;
  const cardYOffset = yBase - peak.height - (baseGap * labelScale);
  const fontSizeMain = baseFontSizeMain * labelScale;
  const fontSizeSub = baseFontSizeSub * labelScale;

  return (
    <g>
      <motion.path
        d={path}
        fill="none"
        stroke="url(#chromaGradient)"
        strokeWidth="3"
        strokeLinecap="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{
          duration: 2,
          ease: "easeInOut",
          delay: peak.delay,
          repeat: Infinity,
          repeatType: "reverse",
          repeatDelay: 4
        }}
      />

      <motion.path
        d={`${path} L ${peak.x + peak.width * 2} ${yBase} L ${peak.x - peak.width * 2} ${yBase} Z`}
        fill="url(#chromaGradient)"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.15 }}
        transition={{
          duration: 2,
          delay: peak.delay,
          repeat: Infinity,
          repeatType: "reverse",
          repeatDelay: 4
        }}
      />

      <motion.g
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        whileHover={{ y: -3, scale: 1.02 }}
        transition={{
          duration: 0.5,
          delay: peak.delay + 1.5,
          repeat: Infinity,
          repeatType: "reverse",
          repeatDelay: 5.5
        }}
        style={{ cursor: 'pointer' }}
      >
        {/* Connector */}
        <line
          x1={peak.x}
          y1={yBase - peak.height - connectorOffset}
          x2={peak.x}
          y2={cardYOffset + cardHeight}
          stroke={labelData.accent}
          strokeWidth="1.5"
          strokeOpacity="0.6"
          strokeDasharray="4 2"
        />
        <circle cx={peak.x} cy={yBase - peak.height - connectorOffset} r={2 * labelScale} fill={labelData.accent} />

        {/* Label Card Group */}
        <g transform={`translate(${peak.x - cardWidth / 2}, ${cardYOffset})`}>
          <rect
            width={cardWidth}
            height={cardHeight}
            rx={cardHeight / 2}
            fill="white"
            fillOpacity="1"
            stroke={labelData.stroke}
            strokeWidth={1 * labelScale}
            filter="url(#glow)"
          />

          <text
            x={cardWidth / 2}
            y={18 * labelScale}
            textAnchor="middle"
            dominantBaseline="middle"
            fill={labelData.color}
            fontSize={fontSizeMain}
            fontWeight="700"
            fontFamily="Inter, sans-serif"
            letterSpacing="-0.01em"
          >
            {labelData.main}
          </text>

          <text
            x={cardWidth / 2}
            y={32 * labelScale}
            textAnchor="middle"
            dominantBaseline="middle"
            fill={labelData.color}
            fillOpacity="0.8"
            fontSize={fontSizeSub}
            fontFamily="Inter, sans-serif"
            fontWeight="500"
          >
            {labelData.sub}
          </text>
        </g>
      </motion.g>
    </g>
  );
};

export default ChromatogramAnimation;
