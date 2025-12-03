import React, { useRef, useEffect } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';

const MoleculeAnimation: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Mouse position state for interaction
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Smooth springs for fluid movement
  const springConfig = { damping: 25, stiffness: 150 };
  const smoothX = useSpring(mouseX, springConfig);
  const smoothY = useSpring(mouseY, springConfig);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        // Calculate relative position (-1 to 1) from center
        const x = (e.clientX - rect.left - rect.width / 2) / (rect.width / 2);
        const y = (e.clientY - rect.top - rect.height / 2) / (rect.height / 2);
        mouseX.set(x);
        mouseY.set(y);
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [mouseX, mouseY]);

  // Generate a mesh of interconnected nodes
  const nodes = [
    { id: 1, x: 20, y: 30, size: 16, color: 'bg-blue-500' },
    { id: 2, x: 50, y: 50, size: 24, color: 'bg-purple-500' },
    { id: 3, x: 80, y: 30, size: 12, color: 'bg-green-500' },
    { id: 4, x: 35, y: 70, size: 20, color: 'bg-pink-500' },
    { id: 5, x: 65, y: 70, size: 14, color: 'bg-yellow-500' },
    { id: 6, x: 50, y: 20, size: 10, color: 'bg-indigo-500' },
  ];

  // Define connections (bonds)
  const bonds = [
    [1, 2], [2, 3], [2, 4], [2, 5], [1, 6], [3, 6]
  ];

  return (
    <div ref={containerRef} className="relative w-full h-[500px] flex items-center justify-center overflow-hidden">
      <motion.div 
        className="relative w-full h-full"
        style={{
          rotateX: useTransform(smoothY, [-1, 1], [10, -10]),
          rotateY: useTransform(smoothX, [-1, 1], [-10, 10]),
          perspective: 1000
        }}
      >
        {/* Bonds (Lines) */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
          {bonds.map(([startId, endId], i) => {
            const start = nodes.find(n => n.id === startId)!;
            const end = nodes.find(n => n.id === endId)!;
            
            // Transform positions based on mouse interaction parallax
            // We can't easily animate SVG coords with Framer springs directly inside map without complex setup,
            // so we'll use CSS transforms on the container or keep lines static relative to nodes for simplicity in this version.
            // For true 3D feel, lines should move with nodes.
            // Simplified: Lines are just static visual connectors in this 2D plane projection
            
            return (
              <line 
                key={i}
                x1={`${start.x}%`} 
                y1={`${start.y}%`} 
                x2={`${end.x}%`} 
                y2={`${end.y}%`} 
                stroke="rgba(148, 163, 184, 0.3)" 
                strokeWidth="2"
              />
            );
          })}
        </svg>

        {/* Nodes (Atoms) */}
        {nodes.map((node) => {
          // Create unique movement factor for parallax depth effect
          const depth = 20 + Math.random() * 30; 
          const moveX = useTransform(smoothX, [-1, 1], [-depth, depth]);
          const moveY = useTransform(smoothY, [-1, 1], [-depth, depth]);

          return (
            <motion.div
              key={node.id}
              className={`absolute rounded-full shadow-lg backdrop-blur-sm ${node.color}`}
              style={{
                left: `${node.x}%`,
                top: `${node.y}%`,
                width: node.size,
                height: node.size,
                x: moveX,
                y: moveY,
              }}
              whileHover={{ scale: 1.2 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }} // Quick hover response
            >
              {/* Inner highlight for 3D sphere look */}
              <div className="absolute top-1 left-1 w-1/3 h-1/3 bg-white/40 rounded-full blur-[1px]" />
            </motion.div>
          );
        })}
      </motion.div>
    </div>
  );
};

export default MoleculeAnimation;
