// Placeholder block with a shimmer band (the existing .sk styles in styles.css; the band is a transform animation).
// Decorative: it is aria-hidden. Put a visible or live status text next to it so screen readers know what is loading.
//   <Skeleton className="sk-title" />                       uses the preset sizes in styles.css
//   <Skeleton width="60%" height={20} radius={8} />        inline size and radius
// Reduced motion removes the band (the grey block stays), see the reduced-motion rules in styles.css.
export default function Skeleton({ width, height, radius, className = '', style, as: Tag = 'span' }) {
  const inline = {
    ...(width !== undefined ? { width } : null),
    ...(height !== undefined ? { height: typeof height === 'number' ? `${height}px` : height } : null),
    ...(radius !== undefined ? { borderRadius: typeof radius === 'number' ? `${radius}px` : radius } : null),
    ...style,
  }
  return <Tag className={`sk skel ${className}`.trim()} style={inline} aria-hidden="true" />
}
