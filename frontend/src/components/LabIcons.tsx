import React from 'react';
import { motion } from 'framer-motion';

export const FlaskIcon = ({ className }: { className?: string }) => (
  <motion.svg 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="1.5" 
    strokeLinecap="round" 
    strokeLinejoin="round" 
    className={className}
    animate={{ rotate: [0, 15, 0, -15, 0] }} // Exaggerated rotation
    transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
  >
    <path d="M10 2v7.31" />
    <path d="M14 2v7.31" />
    <path d="M8.5 2h7" />
    <path d="M14 9.3a6.5 6.5 0 1 1-4 0" />
    <path d="M8.5 22h7" strokeOpacity="0.5" />
    {/* Liquid inside - exaggerated sloshing */}
    <motion.path 
      d="M10 14c0 1.1.9 2 2 2s2-.9 2-2" 
      fill="currentColor" 
      fillOpacity="0.2"
      stroke="none"
      animate={{ scaleY: [1, 1.5, 0.8, 1.5, 1], skewX: [0, 10, 0, -10, 0] }}
      transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
      style={{ originY: 1 }} // Scale from bottom
    />
  </motion.svg>
);

export const BeakerIcon = ({ className }: { className?: string }) => (
  <motion.svg 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="1.5" 
    strokeLinecap="round" 
    strokeLinejoin="round" 
    className={className}
    animate={{ y: [0, -15, 0] }} // Exaggerated floating
    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
  >
    <path d="M4.5 3h15" />
    <path d="M6 3v16a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V3" />
    <path d="M6 14h12" strokeOpacity="0.3" />
    {/* Bubbles - faster and more visible */}
    <motion.circle cx="10" cy="12" r="1.5" fill="currentColor" animate={{ y: -8, opacity: [1, 0] }} transition={{ duration: 1.5, repeat: Infinity }} />
    <motion.circle cx="14" cy="14" r="1.5" fill="currentColor" animate={{ y: -8, opacity: [1, 0] }} transition={{ duration: 2, repeat: Infinity, delay: 0.3 }} />
    <motion.circle cx="8" cy="16" r="1" fill="currentColor" animate={{ y: -6, opacity: [1, 0] }} transition={{ duration: 1.2, repeat: Infinity, delay: 0.7 }} />
  </motion.svg>
);

export const MicroscopeIcon = ({ className }: { className?: string }) => (
  <motion.svg 
    viewBox="0 -4 24 32" // Expanded viewBox to prevent clipping on top/bottom
    fill="none" 
    stroke="currentColor" 
    strokeWidth="1.5" 
    strokeLinecap="round" 
    strokeLinejoin="round" 
    className={className}
    whileHover={{ scale: 1.1 }}
  >
    <path d="M6 18h8" />
    <path d="M3 22h18" />
    <path d="M14 22a7 7 0 1 0 0-14h-1" />
    <path d="M9 14h2" />
    {/* Body moving up and down more exaggeratedly */}
    <motion.path 
      d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z" 
      animate={{ y: [0, -5, 0] }}
      transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
    />
    <motion.path 
      d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3" 
      animate={{ y: [0, -5, 0] }}
      transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
    />
    {/* "Light" or sample scanning effect */}
    <motion.circle 
      cx="10" cy="16" r="2" 
      fill="currentColor" 
      fillOpacity="0.5" 
      animate={{ opacity: [0, 0.8, 0], scale: [0.5, 1.5, 0.5] }} 
      transition={{ duration: 2.5, repeat: Infinity }} 
    />
  </motion.svg>
);

export const DNAIcon = ({ className }: { className?: string }) => (
  <motion.svg 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="1.5" 
    strokeLinecap="round" 
    strokeLinejoin="round" 
    className={className}
    animate={{ rotate: 360 }}
    transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
  >
    <path d="M2 15c6.667-6 13.333 0 20-6" />
    <path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993" />
    <path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993" />
    <path d="M17 6l-2.5-2.5" />
    <path d="M14 8l-1-1" />
    <path d="M7 18l2.5 2.5" />
    <path d="M3.5 14.5l.5.5" />
    <path d="M20 9l.5.5" />
    <path d="M6.5 12.5l1 1" />
    <path d="M16.5 10.5l1 1" />
    <path d="M10 16l1.5 1.5" />
    <path d="M17.5 7.5l-.5-.5" />
  </motion.svg>
);
