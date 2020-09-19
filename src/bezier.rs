use svgtypes::PathSegment;
use ordered_float::NotNan;
use std::ops::{Add, Sub, Mul};

const APPROXIMATION_ERROR: f64 = 0.25; // in pixels
const SPLIT_LIMIT: usize = 1024;

#[derive(Debug, Default, Copy, Clone, PartialEq, PartialOrd)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl std::fmt::Display for Vec2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl Vec2 {
    pub fn lerp(self, ratio: f64, other: Self) -> Self {
        (1.0 * ratio) * self + ratio * other
    }

    pub fn reflect(self, about: Self) -> Self {
        about - (self - about)
    }

    pub fn dot(self, other: Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn sqnorm(self) -> f64 {
        self.dot(self)
    }

    pub fn norm(self) -> f64 {
        self.sqnorm().sqrt()
    }

    pub fn normalized(self) -> Self {
        1.0 / self.norm() * self
    }

    pub fn rotate(self, theta: f64) -> Self {
        Mat2::rotate(theta) * self
    }

    pub fn perp(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    pub fn angle_to(self, other: Self) -> f64 {
        let v1 = self.normalized();
        let v2 = other.normalized();
        return 2.0 * (v1 - v2).norm().atan2(
                     (v1 + v2).norm());
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, other: Self) -> Self { Self { x: self.x + other.x, y: self.y + other.y, } }
}
impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, other: Self) -> Self { Self { x: self.x - other.x, y: self.y - other.y, } }
}
impl Mul<Vec2> for f64 {
    type Output = Vec2;
    fn mul(self, other: Vec2) -> Vec2 { Vec2 { x: self * other.x, y: self * other.y } }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, PartialOrd)]
pub struct Mat2 {
    pub rows: [Vec2; 2],
}

impl Mat2 {
    pub fn rotate(theta: f64) -> Mat2 {
        Mat2 {
            rows: [
                Vec2 { x: theta.cos(), y: -theta.sin() },
                Vec2 { x: theta.sin(), y: theta.cos() },
            ]
        }
    }

    pub fn from_cols(c1: Vec2, c2: Vec2) -> Mat2 {
        Mat2 {
            rows: [
                Vec2 { x: c1.x, y: c2.x },
                Vec2 { x: c1.y, y: c2.y },
            ]
        }
    }

    pub fn transpose(self) -> Mat2 {
        Mat2::from_cols(self.rows[0], self.rows[1])
    }
}

impl Mul<Vec2> for Mat2 {
    type Output = Vec2;
    fn mul(self, v: Vec2) -> Vec2 {
        Vec2 {
            x: self.rows[0].dot(v),
            y: self.rows[1].dot(v),
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct CubicBezier {
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
}

impl CubicBezier {
    fn dot(&self, a0: f64, a1: f64, a2: f64, a3: f64) -> Vec2 {
        a0 * self.p0 + a1 * self.p1 + a2 * self.p2 + a3 * self.p3
    }

    // mat is row-major
    fn matmul(&self, mat: [[f64; 4]; 4]) -> Self {
        CubicBezier {
            p0: self.dot(mat[0][0], mat[0][1], mat[0][2], mat[0][3]),
            p1: self.dot(mat[1][0], mat[1][1], mat[1][2], mat[1][3]),
            p2: self.dot(mat[2][0], mat[2][1], mat[2][2], mat[2][3]),
            p3: self.dot(mat[3][0], mat[3][1], mat[3][2], mat[3][3]),
        }
    }

    fn calculate_split_point(&self, precision: f64) -> f64 {
        // taken from http://www.caffeineowl.com/graphics/2d/vectorial/cubic2quad01.html
        let d01 = 0.5 * (self.p3 - 3.0 * self.p2 + 3.0 * self.p1 - self.p0).norm();
        if d01 < 1e-12 { return 1e12; }
        let k = 18.0 / (3.0f64).sqrt();
        return (k * precision / d01).cbrt();
    }

    fn split(&self, xl: f64, yl: f64) -> Self {
        let xr = 1.0 - xl;
        let yr = 1.0 - yl;
        let mid00 = xr * (yl + xl * (2.0 - 3.0 * yl));
        let mid01 = xl * (xl + yl * (2.0 - 3.0 * xl));
        let mid10 = yr * (xl + yl * (2.0 - 3.0 * xl));
        let mid11 = yl * (yl + xl * (2.0 - 3.0 * yl));
        self.matmul([
            [xr * xr * xr, 3.0 * xr * xr * xl, 3.0 * xr * xl * xl, xl * xl * xl],
            [xr * xr * yr, mid00,              mid01,              xl * xl * yl],
            [xr * yr * yr, mid10,              mid11,              xl * yl * yl],
            [yr * yr * yr, 3.0 * yr * yr * yl, 3.0 * yr * yl * yl, yl * yl * yl],
        ])
    }

    fn to_quadratic_approx(&self) -> Curve {
        let c = 0.25 * (3.0 * self.p2 - self.p3 + 3.0 * self.p1 - self.p0);
        Curve {
            from: self.p0,
            to: self.p3,
            c,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Ellipse {
    c: Vec2,
    rx: f64,
    ry: f64,
    phi: f64,
}

#[derive(Debug, Copy, Clone)]
struct Arc {
    theta1: f64,
    dtheta: f64,
}

impl Ellipse {
    fn arc_from_svg(
            p1: Vec2, p2: Vec2,
            mut rx: f64, mut ry: f64,
            phi: f64,
            large_arc: bool, sweep: bool) -> (Ellipse, Arc) {
        let p1_prime = 0.5 * (p1 - p2).rotate(-phi);
        let lambda = p1_prime.x * p1_prime.x / (rx * rx) +
                     p1_prime.y * p1_prime.y / (ry * ry);
        if lambda >= 1.0 {
            // scale radii
            let sqrt_lambda = lambda.sqrt();
            rx *= sqrt_lambda;
            ry *= sqrt_lambda;
        }
        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let y12 = p1_prime.y * p1_prime.y;
        let x12 = p1_prime.x * p1_prime.x;
        let denom = rx2 * y12 + ry2 * x12;
        let c_prime = (if large_arc == sweep { 1.0 } else { -1.0 }) *
            ((rx2 * ry2 - denom) / denom).sqrt() *
            Vec2 {
                x: rx * p1_prime.y / ry,
                y: -ry * p1_prime.x / rx,
            };
        let c = c_prime.rotate(phi) + 0.5 * (p1 + p2);
        let start = Vec2 {
            x: (p1_prime.x - c_prime.x) / rx,
            y: (p1_prime.y - c_prime.y) / ry
        };
        let end = Vec2 {
            x: (-p1_prime.x - c_prime.x) / rx,
            y: (-p1_prime.y - c_prime.y) / ry
        };
        let theta1 = Vec2 { x: 1.0, y: 0.0 }.angle_to(start);
        let mut dtheta = start.angle_to(end);
        if sweep == false && dtheta > 0.0 {
            dtheta -= 2.0 * std::f64::consts::PI
        } else if sweep == true && dtheta < 0.0 {
            dtheta += 2.0 * std::f64::consts::PI
        }
        (Ellipse {
            c,
            rx,
            ry,
            phi,
        }, Arc {
            theta1,
            dtheta,
        })
    }

    fn estimate_error(&self, arc: Arc) -> f64 {
        // taken from http://www.spaceroots.org/documents/ellipse/node20.html
        let ratio = self.ry / self.rx;
        let eta = arc.theta1 + 0.5 * arc.dtheta;
        let cos0 = 1.0;
        let cos2 = (2.0 * eta).cos();
        let cos4 = (4.0 * eta).cos();
        let cos6 = (6.0 * eta).cos();
        fn model(x: f64, coefs: [f64; 4]) -> f64 {
            (coefs[2] + x * (coefs[1] + x * coefs[0])) / (x + coefs[3])
        }
        const COEFFS_LOW: [[[f64; 4]; 4]; 2] = [
            [
                [ 3.92478,   -13.5822,    -0.233377,    0.0128206  ],
                [-1.08814,     0.859987,   0.000362265, 0.000229036],
                [-0.942512,    0.390456,   0.0080909,   0.00723895 ],
                [-0.736228,    0.20998,    0.0129867,   0.0103456  ],
            ],
            [
                [-0.395018,    6.82464,    0.0995293,   0.0122198  ],
                [-0.545608,    0.0774863,  0.0267327,   0.0132482  ],
                [ 0.0534754,  -0.0884167,  0.012595,    0.0343396  ],
                [ 0.209052,   -0.0599987, -0.00723897,  0.00789976 ],
            ],
        ];
        const COEFFS_HIGH: [[[f64; 4]; 4]; 2] = [
            [
                [ 0.0863805, -11.5595,    -2.68765,    0.181224    ],
                [ 0.242856,   -1.81073,    1.56876,    1.68544     ],
                [ 0.233337,   -0.455621,   0.222856,   0.403469    ],
                [ 0.0612978,  -0.104879,   0.0446799,  0.00867312  ],
            ], [
                [ 0.028973,    6.68407,    0.171472,   0.0211706   ],
                [ 0.0307674,  -0.0517815,  0.0216803, -0.0749348   ],
                [-0.0471179,   0.1288,    -0.0781702,  2.0         ],
                [-0.0309683,   0.0531557, -0.0227191,  0.0434511   ],
            ],
        ];
        const SAFETY: [f64; 4] = [0.02, 2.83, 0.125, 0.01];
        let coeffs = if ratio < 0.25 { &COEFFS_LOW } else { &COEFFS_HIGH };
        let x = ratio;
        let c0 = cos0 * model(x, coeffs[0][0])
               + cos2 * model(x, coeffs[0][1])
               + cos4 * model(x, coeffs[0][2])
               + cos6 + model(x, coeffs[0][3]);
        let c1 = cos0 * model(x, coeffs[1][0])
               + cos2 * model(x, coeffs[1][1])
               + cos4 * model(x, coeffs[1][2])
               + cos6 + model(x, coeffs[1][3]);
        return model(x, SAFETY) * self.rx * (c0 + c1 * arc.dtheta).exp();
    }

    fn at(&self, theta: f64) -> Vec2 {
        self.c + Vec2 { x: self.rx * theta.cos(), y: self.ry * theta.sin() }.rotate(self.phi)
    }

    fn derivative_at(&self, theta: f64) -> Vec2 {
        self.c + Vec2 { x: -self.rx * theta.sin(), y: self.ry * theta.cos() }.rotate(self.phi)
    }

    fn to_quadratic_approx(&self, arc: Arc) -> Curve {
        Curve {
            from: self.at(arc.theta1),
            to: self.at(arc.theta1 + arc.dtheta),
            c: self.at(arc.theta1) + (0.5 * arc.dtheta).tan()
                * self.derivative_at(arc.theta1),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Curve {
    pub from: Vec2,
    pub to: Vec2,
    pub c: Vec2,
}

impl Curve {
    pub fn at(&self, t: f64) -> Vec2 {
        let it = 1.0 - t;
        it * it * self.from + 2.0 * t * it * self.c + t * t * self.to
    }

    pub fn derivative_at(&self, t: f64) -> Vec2 {
        let it = 1.0 - t;
        2.0 * it * (self.c - self.from) + 2.0 * t * (self.to - self.c)
    }

    pub fn hull_distance(&self, p: Vec2) -> f64 {
        fn dist_to_line(u: Vec2, v: Vec2, p: Vec2) -> f64 {
            let l2 = (v - u).sqnorm();
            if l2 < 1e-12 {
                return (p - u).norm();
            }
            let t = (p - u).dot(v - u) / l2;
            let t = t.max(0.0).min(1.0);
            let q = u + t * (v - u);
            (p - q).norm()
        }
        fn point_in_triangle(u: Vec2, v: Vec2, w: Vec2, p: Vec2) -> bool {
            fn sign(u: Vec2, v: Vec2, w: Vec2) -> f64 {
                (u.x - w.x) * (v.y - w.y) - (v.x - w.x) * (u.y - w.y)
            }
            let s1 = sign(p, u, v) > 0.0;
            let s2 = sign(p, v, w) > 0.0;
            if s2 != s1 { return false; }
            let s3 = sign(p, w, u) > 0.0;
            if s3 != s1 { return false; }
            return true;
        }
        if point_in_triangle(self.from, self.to, self.c, p) {
            return 0.0;
        }
        return dist_to_line(self.from, self.to, p)
            .min(dist_to_line(self.to, self.c, p))
            .min(dist_to_line(self.c, self.from, p));
    }

    pub fn ray_crossings(&self, o: Vec2, m: Mat2, ignore_first: bool) -> i32 {
        let rot_p1 = m * (self.from - o);
        let rot_p2 = m * (self.c - o);
        let rot_p3 = m * (self.to - o);
        let (t2_count, t1_count) = {
            let sign_1 = rot_p1.y.is_sign_positive();
            let sign_2 = rot_p2.y.is_sign_positive();
            let sign_3 = rot_p3.y.is_sign_positive();
            match (sign_3, sign_2, sign_1) {
                (false, false, false) => (false, false),
                (false, false, true ) => (false, true ),
                (false, true , false) => (true , true ),
                (false, true , true ) => (false, true ),
                (true , false, false) => (true , false),
                (true , false, true ) => (true , true ),
                (true , true , false) => (true , false),
                (true , true , true ) => (false, false),
            }
        };
        if !t1_count && !t2_count {
            return 0;
        }
        // intersect rotated curve with x axis (y = 0)
        // f + (2 c - 2 f) t + (f - 2 c + t) t^2 = 0
        let v_a = rot_p1 - 2.0 * rot_p2 + rot_p3;
        let v_b = 2.0 * rot_p2 - 2.0 * rot_p1;
        let v_c = rot_p1;
        let a = v_a.y;
        let b = v_b.y;
        let c = v_c.y;
        let disc = b * b - 4.0 * a * c;
        // if tangent or nonreal, will cancel out
        let s = disc.max(0.0).sqrt();
        let (t1, t2) = if a.abs() < 1e-12 {
            (
                -c / b,
                -c / b,
            )
        } else {
            (
                0.5 * (-b - s) / a,
                0.5 * (-b + s) / a,
            )
        };
        let mut x1 = v_a.x * t1 * t1 + v_b.x * t1 + v_c.x;
        let mut x2 = v_a.x * t2 * t2 + v_b.x * t2 + v_c.x;
        if ignore_first {
            if x1.abs() < x2.abs() {
                x1 = -1.0;
            } else {
                x2 = -1.0;
            }
        }
        let mut count = 0;
        if x1 > 0.0 && t1_count {
            count += 1;
        }
        if x2 > 0.0 && t2_count {
            count -= 1;
        }
        count
    }
}

#[derive(Debug, Clone)]
pub struct QuadraticLoop {
    pub start: Vec2,
    pub segments: Vec<Curve>,
}

impl QuadraticLoop {
    pub fn new() -> Self {
        Self {
            start: Vec2 { x: 0.0, y: 0.0 },
            segments: Vec::new(),
        }
    }

    pub fn push_cubic(&mut self, p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) {
        let mut cubic = CubicBezier { p0, p1, p2, p3 };
        let mut rest = Vec::new();
        let mut done = false;
        for _ in 0..SPLIT_LIMIT {
            let t_max = cubic.calculate_split_point(APPROXIMATION_ERROR);
            if t_max >= 1.0 {
                self.segments.push(cubic.to_quadratic_approx());
                done = true;
                break
            } else if t_max >= 0.5 {
                let left  = cubic.split(0.0, 0.5);
                let right = cubic.split(0.5, 1.0);
                self.segments.push(left.to_quadratic_approx());
                self.segments.push(right.to_quadratic_approx());
                done = true;
                break
            }
            let left  = cubic.split(0.0, t_max);
            let mid   = cubic.split(t_max, 1.0 - t_max);
            let right = cubic.split(1.0 - t_max, 1.0);
            self.segments.push(left.to_quadratic_approx());
            rest.push(right.to_quadratic_approx());
            cubic = mid;
        }
        if !done {
            // should be rare as approximations are pretty good
            self.segments.push(cubic.to_quadratic_approx());
        }
        for seg in rest.into_iter().rev() {
            self.segments.push(seg);
        }
    }

    pub fn push_arc(&mut self,
                    p1: Vec2, p2: Vec2,
                    rx: f64, ry: f64,
                    phi: f64,
                    large_arc: bool, sweep: bool) {
        let (ell, mut arc) = Ellipse::arc_from_svg(
            p1, p2, rx, ry, phi, large_arc, sweep);
        let mut rest = Vec::new();
        loop {
            if ell.estimate_error(arc) < APPROXIMATION_ERROR {
                self.segments.push(ell.to_quadratic_approx(arc));
                arc = match rest.pop() {
                    Some(next) => next,
                    None => break,
                };
                continue;
            }
            // subdivide
            rest.push(Arc { theta1: arc.theta1 + 0.5 * arc.dtheta, dtheta: 0.5 * arc.dtheta });
            arc.dtheta *= 0.5;
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuadraticPath {
    pub loops: Vec<QuadraticLoop>,
}

impl QuadraticPath {
    pub fn from_svg_path<P: Iterator<Item=PathSegment>>(path: P) -> Self {
        fn rel_to_abs(cur: Vec2, abs: bool, x: f64, y: f64) -> Vec2 {
            if abs {
                Vec2 { x, y }
            } else {
                Vec2 { x, y } + cur
            }
        }

        let mut out = QuadraticPath { loops: Vec::new() };
        let mut cur = Vec2 { x: 0.0, y: 0.0 };
        let mut curloop = QuadraticLoop::new();
        let mut last_quad = cur;
        let mut last_cubic = cur;

        fn close(out: &mut QuadraticPath, curloop: &mut QuadraticLoop, cur: Vec2) {
            use std::mem::replace;
            if curloop.segments.len() == 0 { return; }
            if cur.x != curloop.start.x || cur.y != curloop.start.y {
                curloop.segments.push(Curve {
                    from: cur,
                    to: curloop.start,
                    c: cur.lerp(0.5, curloop.start),
                });
            }
            let old = replace(curloop, QuadraticLoop::new());
            out.loops.push(old);
        }

        for segment in path {
            match segment {
                PathSegment::ClosePath { abs: _ } => {
                    let start = curloop.start;
                    close(&mut out, &mut curloop, cur);
                    cur = start;
                    last_quad = cur;
                    last_cubic = cur;
                },
                PathSegment::MoveTo { abs, x, y } => {
                    close(&mut out, &mut curloop, cur);
                    if abs {
                        cur.x = x;
                        cur.y = y;
                    } else {
                        cur.x += x;
                        cur.y += y;
                    }
                    curloop.start = cur;
                    last_quad = cur;
                    last_cubic = cur;
                },
                PathSegment::LineTo { abs, .. }
                | PathSegment::HorizontalLineTo { abs, .. }
                | PathSegment::VerticalLineTo { abs, .. } => {
                    let x = segment.x();
                    let y = segment.y();
                    let to = if abs {
                        Vec2 { x: x.unwrap_or(cur.x), y: y.unwrap_or(cur.y) }
                    } else {
                        Vec2 { x: cur.x + x.unwrap_or(0.0), y: cur.y + y.unwrap_or(0.0) }
                    };
                    curloop.segments.push(Curve {
                        from: cur,
                        to,
                        c: cur.lerp(0.5, to),
                    });
                    cur = to;
                    last_quad = cur;
                    last_cubic = cur;
                },
                PathSegment::CurveTo { abs, x, y, x2, y2, .. }
                | PathSegment::SmoothCurveTo { abs, x, y, x2, y2, .. } => {
                    let to = rel_to_abs(cur, abs, x, y);
                    let c2 = rel_to_abs(cur, abs, x2, y2);
                    let c1 = match segment {
                        PathSegment::CurveTo { x1, y1, .. } =>
                            rel_to_abs(cur, abs, x1, y1),
                        _ => last_cubic.reflect(cur),
                    };
                    curloop.push_cubic(cur, c1, c2, to);
                    cur = to;
                    last_quad = cur;
                    last_cubic = c2;
                },
                PathSegment::Quadratic { abs, x, y, .. }
                | PathSegment::SmoothQuadratic { abs, x, y } => {
                    let to = rel_to_abs(cur, abs, x, y);
                    let c = match segment {
                        PathSegment::Quadratic { x1, y1, .. } =>
                            rel_to_abs(cur, abs, x1, y1),
                        _ => last_quad.reflect(cur),
                    };
                    curloop.segments.push(Curve {
                        from: cur,
                        to,
                        c,
                    });
                    cur = to;
                    last_quad = c;
                    last_cubic = cur;
                },
                PathSegment::EllipticalArc {
                    abs,
                    rx,
                    ry,
                    x_axis_rotation,
                    large_arc,
                    sweep,
                    x,
                    y,
                } => {
                    let to = rel_to_abs(cur, abs, x, y);
                    if to.x == cur.x && to.y == cur.y {
                        continue;
                    }
                    let rx = rx.abs();
                    let ry = ry.abs();
                    if rx == 0.0 || ry == 0.0 {
                        curloop.segments.push(Curve {
                            from: cur,
                            to,
                            c: cur.lerp(0.5, to),
                        });
                        continue;
                    }
                    curloop.push_arc(cur, to, rx, ry, x_axis_rotation.to_radians(), large_arc, sweep);
                }
            }
        }
        close(&mut out, &mut curloop, cur);
        out
    }

    // returns (normal, inside, distance)
    pub fn calculate_true_normal(&self, p: Vec2) -> (Vec2, bool, f64) {
        use crate::FIX_THRESHOLD;
        // first, find the curve it lies on
        let mut best_dist2 = std::f64::INFINITY;
        let mut curve_index = None;
        let mut normal = Vec2 { x: 0.0, y: 0.0 };
        let mut curve_point = Vec2 { x: 0.0, y: 0.0 };
        for (i, l) in self.loops.iter().enumerate() {
            for (j, curve) in l.segments.iter().enumerate() {
                let min_dist = curve.hull_distance(p);
                if min_dist > 2.0 * FIX_THRESHOLD as f64 {
                    continue;
                }
                // compute cubic coefficients
                let va = curve.from - p;
                let vb = -2.0 * curve.from + 2.0 * curve.c;
                let vc = curve.from - 2.0 * curve.c + curve.to;
                let a = 4.0 * vc.sqnorm();
                let b = 6.0 * vb.dot(vc);
                let c = 2.0 * (2.0 * va.dot(vc) + vb.sqnorm());
                let d = 2.0 * va.dot(vb);
                let solns = solve_cubic(a, b, c, d);
                let best_t = solns.into_iter()
                    .map(|t| newton_iterate(a, b, c, d, t))
                    .map(|t| newton_iterate(a, b, c, d, t))
                    .filter(|&t| t > 0.0 && t < 1.0)
                    .chain(vec![0.0, 1.0])
                    .min_by_key(|&t| NotNan::new((curve.at(t) - p).sqnorm()).unwrap())
                    .unwrap();
                let dist2 = (curve.at(best_t) - p).sqnorm();
                if dist2 < best_dist2 {
                    best_dist2 = dist2;
                    curve_index = Some((i, j));
                    normal = curve.derivative_at(best_t).perp();
                    curve_point = curve.at(best_t);
                }
            }
        }
        // to choose the correct normal, fire a ray starting from
        // p + epsilon * normal
        let curve_index = curve_index.expect("no curve close to point");
        let numerical_normal = p - curve_point;
        normal = 1.0 / normal.norm() * normal;
        if normal.dot(numerical_normal) < 0.0 {
            normal = -1.0 * normal;
        }
        let transform = Mat2::from_cols(normal, normal.perp()).transpose();
        // calculate winding number
        let mut winding_number: i32 = 0;
        for (i, l) in self.loops.iter().enumerate() {
            for (j, c) in l.segments.iter().enumerate() {
                // to avoid floating-point errors, ignore the original curve
                // note: since normal points in the same direction
                // as numerical_normal, it never lies on the ray
                let ignore = (i, j) == curve_index;
                let num_crossings = c.ray_crossings(p, transform, ignore);
                winding_number += num_crossings;
            }
        }
        // if normal points inside, flip it
        let inside = winding_number != 0;
        if inside {
            (-1.0 * normal, true, best_dist2.sqrt())
        } else {
            (normal, false, best_dist2.sqrt())
        }
    }
}

// copied from GLSL
fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    // solves a x^3 + b x^2 + c x + d = 0
    if a.abs() < 1e-12 {
        if b.abs() < 1e-12 {
            if c.abs() < 1e-12 {
                // degenerate case
                if d.abs() < 1e-12 {
                    return vec![0.0];
                }
                return vec![];
            }
            return vec![-d / c];
        }
        // quadratic case
        // x = -c +- sqrt(c^2 - 4 b d) / 2 b
        let disc = c * c - 4.0 * b * d;
        let s = disc.max(0.0).sqrt();
        if disc > -1e-12 {
            return vec![
                0.5 * (-c + s) / b,
                0.5 * (-c - s) / b,
            ];
        } else {
            return vec![];
        }
    }
    // cubic case
    // depress to form - x^3 + 3 p x + 2 q = 0
    let p = -(3.0 * a * c - b * b) / (9.0 * a * a);
    let q = -(2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (54.0 * a * a * a);
    let off = -b / (3.0 * a);
    if q.abs() < 1e-12 {
        // -x^3 + 3 p x = 0
        // x (-x^2 + 3 p) = 0
        // x = 0, or
        // x = +-sqrt(3 p)
        let s = (3.0 * p).max(0.0).sqrt();
        if p > -1e-12 {
            return vec![off, off + s, off - s];
        } else {
            return vec![off];
        }
    } else {
        let w = -q * q;
        let e = (-q).mul_add(q, -w);
        let s = p * p;
        let t = p.mul_add(p, -s);
        let f = s.mul_add(p, w);
        let u = t * p;
        let v = t.mul_add(p, -u);
        let x_a = u + f;
        let x_b = x_a - u;
        let x_c = x_a - x_b;
        let x_b = f - x_b;
        let x_c = u - x_c;
        let p3mq2 = (((e + x_a) + x_b) + x_c) + v;
        if p3mq2 < -1e-12 {
            // one real root
            if p > 0.0 {
                return vec![
                    off + 2.0 * q.signum() * p.sqrt() *
                        ((q.abs() / p * (1.0 / p).sqrt()).acosh() / 3.0).cosh()
                ];
            } else if p < 0.0 {
                return vec![
                    off - 2.0 * (-p).sqrt() *
                        ((q / p * (-1.0 / p).sqrt()).asinh() / 3.0).sinh()
                ]
            } else {
                // -x^3 + 2 q = 0
                // x = cbrt(2 q)
                return vec![(2.0 * q).cbrt()];
            }
        } else {
            // three real roots
            let theta = p3mq2.max(0.0).sqrt().atan2(q) / 3.0;
            let sp = p.max(0.0).sqrt();
            let pi3 = std::f64::consts::FRAC_PI_3;
            return vec![
                off + 2.0 * sp * (theta).cos(),
                off + 2.0 * sp * (theta + 2.0 * pi3).cos(),
                off + 2.0 * sp * (theta + 4.0 * pi3).cos(),
            ];
        }
    }
}


fn newton_iterate(a: f64, b: f64, c: f64, d: f64, x: f64) -> f64 {
    let y = a * x * x * x + b * x * x + c * x + d;
    let dy = 3.0 * a * x * x + 2.0 * b * x + c;
    x - y / dy
}
