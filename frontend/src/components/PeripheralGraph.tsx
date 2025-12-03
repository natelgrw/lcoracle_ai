import React, { useEffect, useRef } from 'react';

interface Node {
  id: number;
  x: number; // Percentage 0-100
  y: number; // Percentage 0-100
  vx: number;
  vy: number;
  size: number;
  color: string;
}

const PeripheralGraph: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Configuration
  const connectionDistance = 15; // Percentage distance
  const mouseRadius = 20; // Percentage radius
  const edgeBuffer = 2; // Percentage buffer from edges

  const colors = ['#3b82f6', '#ef4444', '#22c55e', '#eab308', '#a855f7', '#ec4899'];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let nodes: Node[] = [];
    let mouseX = -1000;
    let mouseY = -1000;
    let animationFrameId: number;

    const initNodes = () => {
      const width = window.innerWidth;
      let nodeCount = 15;
      if (width > 1200) nodeCount = 35; // More nodes on wide screens
      else if (width > 768) nodeCount = 20; // Medium
      else nodeCount = 12; // Mobile

      nodes = [];
      const zones = [
        { xMin: edgeBuffer, xMax: 25, yMin: edgeBuffer, yMax: 100 - edgeBuffer },     // Left Side
        { xMin: 75, xMax: 100 - edgeBuffer, yMin: edgeBuffer, yMax: 100 - edgeBuffer },   // Right Side
      ];

      for (let i = 0; i < nodeCount; i++) {
        const zone = zones[Math.floor(Math.random() * zones.length)];
        nodes.push({
          id: i,
          x: zone.xMin + Math.random() * (zone.xMax - zone.xMin),
          y: zone.yMin + Math.random() * (zone.yMax - zone.yMin),
          vx: (Math.random() - 0.5) * 0.05, // Keeping user's preferred speed
          vy: (Math.random() - 0.5) * 0.05,
          size: 2 + Math.random() * 3,
          color: colors[Math.floor(Math.random() * colors.length)],
        });
      }
    };

    // Initial init
    initNodes();

    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
      }
      // Re-init nodes on significant resize to adjust count
      initNodes();
    };
    window.addEventListener('resize', resize);
    resize();

    const animate = () => {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Physics update & Draw
      nodes.forEach(node => {
        // Move
        node.x += node.vx;
        node.y += node.vy;

        // Mouse interaction
        const mpX = (mouseX / canvas.width) * 100;
        const mpY = (mouseY / canvas.height) * 100;
        const dx = node.x - mpX;
        const dy = node.y - mpY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < mouseRadius) {
          // Repel
          const force = (mouseRadius - dist) / mouseRadius;
          node.vx += (dx / dist) * force * 0.5;
          node.vy += (dy / dist) * force * 0.5;
        }

        // Limit Max Velocity
        const maxV = 0.1;
        if (node.vx > maxV) node.vx = maxV;
        if (node.vx < -maxV) node.vx = -maxV;
        if (node.vy > maxV) node.vy = maxV;
        if (node.vy < -maxV) node.vy = -maxV;

        // Hard Boundary Constraint (Bounce)
        if (node.x <= edgeBuffer) {
          node.x = edgeBuffer;
          node.vx = Math.abs(node.vx); // Force positive
        } else if (node.x >= 100 - edgeBuffer) {
          node.x = 100 - edgeBuffer;
          node.vx = -Math.abs(node.vx); // Force negative
        }

        if (node.y <= edgeBuffer) {
          node.y = edgeBuffer;
          node.vy = Math.abs(node.vy);
        } else if (node.y >= 100 - edgeBuffer) {
          node.y = 100 - edgeBuffer;
          node.vy = -Math.abs(node.vy);
        }

        // Keep out of center safety zone (25% to 75% X) - Soft constraint
        if (node.x > 25 && node.x < 75) {
          if (node.x < 50) node.vx -= 0.0005;
          else node.vx += 0.0005;
        }

        // Draw connections
        const px = (node.x / 100) * canvas.width;
        const py = (node.y / 100) * canvas.height;

        nodes.forEach(other => {
          if (node.id === other.id) return;
          const dx = node.x - other.x;
          const dy = node.y - other.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < connectionDistance) {
            const opx = (other.x / 100) * canvas.width;
            const opy = (other.y / 100) * canvas.height;

            const opacity = 1 - (dist / connectionDistance);
            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(opx, opy);

            const grad = ctx.createLinearGradient(px, py, opx, opy);
            grad.addColorStop(0, node.color);
            grad.addColorStop(1, other.color);

            ctx.strokeStyle = grad;
            ctx.lineWidth = 1 * opacity;
            ctx.globalAlpha = opacity * 0.6;
            ctx.stroke();
            ctx.globalAlpha = 1;
          }
        });
      });

      // Draw nodes on top
      nodes.forEach(node => {
        const px = (node.x / 100) * canvas.width;
        const py = (node.y / 100) * canvas.height;

        ctx.shadowBlur = 10;
        ctx.shadowColor = node.color;

        ctx.beginPath();
        ctx.arc(px, py, node.size, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.fill();

        ctx.shadowBlur = 0;
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseX = e.clientX - rect.left;
      mouseY = e.clientY - rect.top;
    };

    window.addEventListener('mousemove', handleMouseMove);
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

export default PeripheralGraph;
