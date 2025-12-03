import React, { useEffect, useRef } from 'react';

const HexagonBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let mouseX = -1000;
    let mouseY = -1000;
    
    // Configuration
    const hexRadius = 25; 
    const hexGap = 2;
    const baseOpacity = 0.008; // Much fainter
    const highlightOpacity = 0.4;
    const interactionRadius = 150; // Smaller range
    
    // Calculated values
    const a = (2 * Math.PI) / 6;
    const r = hexRadius;
    const w = r * 2; // width of hex (point to point)
    const h = Math.sqrt(3) * r; // height of hex (flat to flat)
    // Grid spacing (pointy topped)
    // Horizontal distance between centers: w * 3/4
    // Vertical distance between centers: h
    const xDist = r * 1.5 + hexGap; // Horizontal distance between adjacent columns
    const yDist = h + hexGap; // Vertical distance between rows

    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
      }
    };
    
    // Initial resize
    resize();
    window.addEventListener('resize', resize);

    const drawHexagon = (x: number, y: number, r: number, opacity: number) => {
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        ctx.lineTo(x + r * Math.cos(a * i), y + r * Math.sin(a * i));
      }
      ctx.closePath();
      ctx.strokeStyle = `rgba(59, 130, 246, ${opacity})`; // blue-500 based
      ctx.lineWidth = 1;
      ctx.stroke();
      
      // Optional: fill slightly
      if (opacity > baseOpacity + 0.05) {
          ctx.fillStyle = `rgba(59, 130, 246, ${opacity * 0.1})`;
          ctx.fill();
      }
    };

    const animate = () => {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const cols = Math.ceil(canvas.width / xDist) + 2;
      const rows = Math.ceil(canvas.height / yDist) + 2;

      for (let col = -1; col < cols; col++) {
        for (let row = -1; row < rows; row++) {
          let x = col * xDist;
          let y = row * yDist;

          // Offset every other column
          if (col % 2 !== 0) {
            y += yDist / 2;
          }

          // Calculate distance to mouse
          const dx = x - mouseX;
          const dy = y - mouseY;
          const dist = Math.sqrt(dx * dx + dy * dy);

          let opacity = baseOpacity;
          let currentRadius = r;

          if (dist < interactionRadius) {
            // Calculate intensity (0 to 1) based on distance
            const intensity = 1 - Math.pow(dist / interactionRadius, 2);
            opacity = baseOpacity + (highlightOpacity - baseOpacity) * intensity;
            // Subtle scale effect
            currentRadius = r + (intensity * 2); 
          }

          drawHexagon(x, y, currentRadius, opacity);
        }
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseX = e.clientX - rect.left;
      mouseY = e.clientY - rect.top;
    };

    const handleMouseLeave = () => {
      mouseX = -1000;
      mouseY = -1000;
    };

    window.addEventListener('mousemove', handleMouseMove); // Window listener for smoother tracking across elements
    
    animate();

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      className="absolute inset-0 z-0 pointer-events-none" 
      style={{ width: '100%', height: '100%' }}
    />
  );
};

export default HexagonBackground;

