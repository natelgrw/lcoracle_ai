import React, { useRef, useEffect, useState } from 'react';
import { useAnimationFrame } from 'framer-motion';

interface Node {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  color: string;
}

const FloatingGraph: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // initialize nodes
  useEffect(() => {
    if (!containerRef.current) return;

    const { clientWidth, clientHeight } = containerRef.current;
    setDimensions({ width: clientWidth, height: clientHeight });

    const colors = ['bg-blue-500', 'bg-indigo-500', 'bg-purple-500', 'bg-cyan-500'];

    const initialNodes = Array.from({ length: 25 }).map((_, i) => ({
      id: i,
      x: Math.random() * clientWidth,
      y: Math.random() * clientHeight,
      vx: (Math.random() - 0.5) * 1.5,
      vy: (Math.random() - 0.5) * 1.5,
      radius: 4 + Math.random() * 8,
      color: colors[Math.floor(Math.random() * colors.length)]
    }));
    setNodes(initialNodes);
  }, []);

  // update loop
  useAnimationFrame(() => {
    if (!dimensions.width) return;

    setNodes(prevNodes => prevNodes.map(node => {
      let { x, y, vx, vy } = node;

      // repulsive mouse interaction of nodes
      const dx = x - mousePos.x;
      const dy = y - mousePos.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const repulsionRadius = 200;

      if (dist < repulsionRadius) {
        const force = (repulsionRadius - dist) / repulsionRadius;
        vx += (dx / dist) * force * 0.8;
        vy += (dy / dist) * force * 0.8;
      }

      x += vx;
      y += vy;

      const boundaryPadding = 50;
      if (x < -boundaryPadding || x > dimensions.width + boundaryPadding) vx *= -1;
      if (y < -boundaryPadding || y > dimensions.height + boundaryPadding) vy *= -1;

      const maxSpeed = 3;
      const speed = Math.sqrt(vx * vx + vy * vy);
      if (speed > maxSpeed) {
        vx = (vx / speed) * maxSpeed;
        vy = (vy / speed) * maxSpeed;
      }

      // constant motion of nodes
      if (speed < 0.5) {
        vx *= 1.05;
        vy *= 1.05;
      }

      return { ...node, x, y, vx, vy };
    }));
  });

  const handleMouseMove = (e: React.MouseEvent) => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    }
  };

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      className="relative w-full h-full min-h-[500px] overflow-hidden bg-transparent"
    >
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        {/* Draw connections */}
        {nodes.map((nodeA, i) =>
          nodes.slice(i + 1).map((nodeB) => {
            const dx = nodeA.x - nodeB.x;
            const dy = nodeA.y - nodeB.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            // connect nodes with edges if close enough
            if (dist < 200) {
              const opacity = 1 - dist / 200;
              return (
                <line
                  key={`${nodeA.id}-${nodeB.id}`}
                  x1={nodeA.x}
                  y1={nodeA.y}
                  x2={nodeB.x}
                  y2={nodeB.y}
                  stroke={`rgba(99, 102, 241, ${opacity * 0.3})`}
                  strokeWidth="1.5"
                />
              );
            }
            return null;
          })
        )}
      </svg>

      {/* Render Nodes */}
      {nodes.map(node => (
        <div
          key={node.id}
          className={`absolute rounded-full ${node.color} shadow-[0_0_15px_rgba(99,102,241,0.4)] backdrop-blur-sm`}
          style={{
            left: node.x,
            top: node.y,
            width: node.radius * 2,
            height: node.radius * 2,
            transform: 'translate(-50%, -50%)',
          }}
        />
      ))}
    </div>
  );
};

export default FloatingGraph;
