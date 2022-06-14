use std::cmp::Reverse;
use std::collections::{BTree, BinaryHeap};
use std::ops::Add;

fn PlayerMovement(){
    type Graph<V, E> = BTreeMap<V, BTreeMap<V, E>>;

fn add_edge<V: Ord + Copy, E: Ord + Add + Copy>(graph: &mut Graph<V, E>, v1: V, v2: V, c: E) {
    graph.entry(v1).or_insert_with(BTreeMap::new).insert(v2, c);
    graph.entry(v2).or_insert_with(BTreeMap::new).insert(v1, c);
}


pub fn prim<V: Ord + Copy + std::fmt::Debug, E: Ord + Add + Copy + std::fmt::Debug>(
    graph: &Graph<V, E>,
) -> Graph<V, E> {
    match graph.keys().next() {
        Some(v) => prim_with_start(graph, *v),
        None => BTreeMap::new(),
    }
}

pub fn prim_with_start<V: Ord + Copy, E: Ord + Add + Copy>(
    graph: &Graph<V, E>,
    start: V,
) -> Graph<V, E> {
    
    let mut mst: Graph<V, E> = Graph::new();
    
    
    let mut prio = BinaryHeap::new();

    mst.insert(start, BTreeMap::new());

    for (v, c) in &graph[&start] {
        prio.push(Reverse((*c, v, start)));
    }

    while let Some(Reverse((dist, t, prev))) = prio.pop() {
        if mst.contains_key(t) {
            continue;
        }

        add_edge(&mut mst, prev, *t, dist);

        for (v, c) in &graph[t] {
            if !mst.contains_key(v) {
                prio.push(Reverse((*c, v, *t)));
            }
        }
    }

    mst
}

#[cfg(test)]
mod tests {
    use super::{add_edge, prim, Graph};
    use std::collections::BTreeMap;

    #[test]
    fn empty() {
        assert_eq!(prim::<usize, usize>(&BTreeMap::new()), BTreeMap::new());
    }

    #[test]
    fn single_vertex() {
        let mut graph: Graph<usize, usize> = BTreeMap::new();
        graph.insert(42, BTreeMap::new());

        assert_eq!(prim(&graph), graph);
    }

    #[test]
    fn single_edge() {
        let mut graph = BTreeMap::new();

        add_edge(&mut graph, 42, 666, 12);

        assert_eq!(prim(&graph), graph);
    }

    #[test]
    fn tree_1() {
        let mut graph = BTreeMap::new();

        add_edge(&mut graph, 0, 1, 10);
        add_edge(&mut graph, 0, 2, 11);
        add_edge(&mut graph, 2, 3, 12);
        add_edge(&mut graph, 2, 4, 13);
        add_edge(&mut graph, 1, 5, 14);
        add_edge(&mut graph, 1, 6, 15);
        add_edge(&mut graph, 3, 7, 16);

        assert_eq!(prim(&graph), graph);
    }

    #[test]
    fn tree_2() {
        let mut graph = BTreeMap::new();

        add_edge(&mut graph, 1, 2, 11);
        add_edge(&mut graph, 2, 3, 12);
        add_edge(&mut graph, 2, 4, 13);
        add_edge(&mut graph, 4, 5, 14);
        add_edge(&mut graph, 4, 6, 15);
        add_edge(&mut graph, 6, 7, 16);

        assert_eq!(prim(&graph), graph);
    }

    #[test]
    fn tree_3() {
        let mut graph = BTreeMap::new();

        for i in 1..100 {
            add_edge(&mut graph, i, 2 * i, i);
            add_edge(&mut graph, i, 2 * i + 1, -i);
        }

        assert_eq!(prim(&graph), graph);
    }

    #[test]
    fn graph_1() {
        let mut graph = BTreeMap::new();
        add_edge(&mut graph, 'a', 'b', 6);
        add_edge(&mut graph, 'a', 'c', 7);
        add_edge(&mut graph, 'a', 'e', 2);
        add_edge(&mut graph, 'a', 'f', 3);
        add_edge(&mut graph, 'b', 'c', 5);
        add_edge(&mut graph, 'c', 'e', 5);
        add_edge(&mut graph, 'd', 'e', 4);
        add_edge(&mut graph, 'd', 'f', 1);
        add_edge(&mut graph, 'e', 'f', 2);

        let mut ans = BTreeMap::new();
        add_edge(&mut ans, 'd', 'f', 1);
        add_edge(&mut ans, 'e', 'f', 2);
        add_edge(&mut ans, 'a', 'e', 2);
        add_edge(&mut ans, 'b', 'c', 5);
        add_edge(&mut ans, 'c', 'e', 5);

        assert_eq!(prim(&graph), ans);
    }

    #[test]
    fn graph_2() {
        let mut graph = BTreeMap::new();
        add_edge(&mut graph, 1, 2, 6);
        add_edge(&mut graph, 1, 3, 1);
        add_edge(&mut graph, 1, 4, 5);
        add_edge(&mut graph, 2, 3, 5);
        add_edge(&mut graph, 2, 5, 3);
        add_edge(&mut graph, 3, 4, 5);
        add_edge(&mut graph, 3, 5, 6);
        add_edge(&mut graph, 3, 6, 4);
        add_edge(&mut graph, 4, 6, 2);
        add_edge(&mut graph, 5, 6, 6);

        let mut ans = BTreeMap::new();
        add_edge(&mut ans, 1, 3, 1);
        add_edge(&mut ans, 4, 6, 2);
        add_edge(&mut ans, 2, 5, 3);
        add_edge(&mut ans, 2, 3, 5);
        add_edge(&mut ans, 3, 6, 4);

        assert_eq!(prim(&graph), ans);
    }

    #[test]
    fn graph_3() {
        let mut graph = BTreeMap::new();
        add_edge(&mut graph, "v1", "v2", 1);
        add_edge(&mut graph, "v1", "v3", 3);
        add_edge(&mut graph, "v1", "v5", 6);
        add_edge(&mut graph, "v2", "v3", 2);
        add_edge(&mut graph, "v2", "v4", 3);
        add_edge(&mut graph, "v2", "v5", 5);
        add_edge(&mut graph, "v3", "v4", 5);
        add_edge(&mut graph, "v3", "v6", 2);
        add_edge(&mut graph, "v4", "v5", 2);
        add_edge(&mut graph, "v4", "v6", 4);
        add_edge(&mut graph, "v5", "v6", 1);

        let mut ans = BTreeMap::new();
        add_edge(&mut ans, "v1", "v2", 1);
        add_edge(&mut ans, "v5", "v6", 1);
        add_edge(&mut ans, "v2", "v3", 2);
        add_edge(&mut ans, "v3", "v6", 2);
        add_edge(&mut ans, "v4", "v5", 2);

        assert_eq!(prim(&graph), ans);
    }
}
}

fn CreateNewLevel(){
    type Point = (f64, f64);
    type LevelGeo = Point;
    type LevelSoup = LevelGeo;
    type LevelMod = LevelSoup;
    type LevelDirectory = LevelMod;
    use std::cmp::Ordering;
    
    fn point_cmp((a1, a2): &Point, (b1, b2): &Point) -> Ordering {
        let acmp = f64_cmp(a1, b1);
        match acmp {
            Ordering::Equal => f64_cmp(a2, b2),
            _ => acmp,
        }
    }
    
    fn f64_cmp(a: &f64, b: &f64) -> Ordering {
        a.partial_cmp(b).unwrap()
    }

    pub fn closest_points(points: &[Point]) -> Option<(Point, Point)> {
        let mut points: Vec<Point> = points.to_vec();
        points.sort_by(point_cmp);
    
        closest_points_aux(&points, 0, points.len())
    }
    
    fn dist((x1, y1): &Point, (x2, y2): &Point) -> f64 {
        let dx = *x1 - *x2;
        let dy = *y1 - *y2;
    
        (dx * dx + dy * dy).sqrt()
    }
    
    fn closest_points_aux(
        points: &[Point],
        mut start: usize,
        mut end: usize,
    ) -> Option<(Point, Point)> {
        let n = end - start;
    
        if n <= 1 {
            return None;
        }
    
        if n <= 3 {
            
            let mut min = dist(&points[0], &points[1]);
            let mut pair = (points[0], points[1]);
    
            for i in 0..n {
                for j in (i + 1)..n {
                    let new = dist(&points[i], &points[j]);
                    if new < min {
                        min = new;
                        pair = (points[i], points[j]);
                    }
                }
            }
            return Some(pair);
        }
    
        let mid = (start + end) / 2;
        let left = closest_points_aux(points, start, mid);
        let right = closest_points_aux(points, mid, end);
    
        let (mut min_dist, mut pair) = match (left, right) {
            (Some((l1, l2)), Some((r1, r2))) => {
                let dl = dist(&l1, &l2);
                let dr = dist(&r1, &r2);
                if dl < dr {
                    (dl, (l1, l2))
                } else {
                    (dr, (r1, r2))
                }
            }
            (Some((a, b)), None) => (dist(&a, &b), (a, b)),
            (None, Some((a, b))) => (dist(&a, &b), (a, b)),
            (None, None) => unreachable!(),
        };
    
        let mid_x = points[mid].0;
        while points[start].0 < mid_x - min_dist {
            start += 1;
        }
        while points[end - 1].0 > mid_x + min_dist {
            end -= 1;
        }
    
        let mut mids: Vec<&Point> = points[start..end].iter().collect();
        mids.sort_by(|a, b| f64_cmp(&a.1, &b.1));
    
        for (i, e) in mids.iter().enumerate() {
            for k in 1..8 {
                if i + k >= mids.len() {
                    break;
                }
    
                let new = dist(e, mids[i + k]);
                if new < min_dist {
                    min_dist = new;
                    pair = (**e, *mids[i + k]);
                }
            }
        }
    
        Some(pair)
    }
    
    #[cfg(test)]
    mod tests {
        use super::closest_points;
        use super::Point;
    
        fn eq(p1: Option<(Point, Point)>, p2: Option<(Point, Point)>) -> bool {
            match (p1, p2) {
                (None, None) => true,
                (Some((p1, p2)), Some((p3, p4))) => (p1 == p3 && p2 == p4) || (p1 == p4 && p2 == p3),
                _ => false,
            }
        }
    
        macro_rules! assert_display {
            ($left: expr, $right: expr) => {
                assert!(
                    eq($left, $right),
                    "assertion failed: `(left == right)`\nleft: `{:?}`,\nright: `{:?}`",
                    $left,
                    $right
                )
            };
        }
    
        #[test]
        fn zero_points() {
            let vals: [Point; 0] = [];
            assert_display!(closest_points(&vals), None::<(Point, Point)>);
        }
    
        #[test]
        fn one_points() {
            let vals = [(0., 0.)];
            assert_display!(closest_points(&vals), None::<(Point, Point)>);
        }
    
        #[test]
        fn two_points() {
            let vals = [(0., 0.), (1., 1.)];
            assert_display!(closest_points(&vals), Some(((0., 0.), (1., 1.))));
        }
    
        #[test]
        fn three_points() {
            let vals = [(0., 0.), (1., 1.), (3., 3.)];
            assert_display!(closest_points(&vals), Some(((0., 0.), (1., 1.))));
        }
    
        #[test]
        fn list_1() {
            let vals = [
                (0., 0.),
                (2., 1.),
                (5., 2.),
                (2., 3.),
                (4., 0.),
                (0., 4.),
                (5., 6.),
                (4., 4.),
                (7., 3.),
                (-1., 2.),
                (2., 6.),
            ];
            assert_display!(closest_points(&vals), Some(((2., 1.), (2., 3.))));
        }
    
        #[test]
        fn list_2() {
            let vals = [
                (1., 3.),
                (4., 6.),
                (8., 8.),
                (7., 5.),
                (5., 3.),
                (10., 3.),
                (7., 1.),
                (8., 3.),
                (4., 9.),
                (4., 12.),
                (4., 15.),
                (7., 14.),
                (8., 12.),
                (6., 10.),
                (4., 14.),
                (2., 7.),
                (3., 8.),
                (5., 8.),
                (6., 7.),
                (8., 10.),
                (6., 12.),
            ];
            assert_display!(closest_points(&vals), Some(((4., 14.), (4., 15.))));
        }
    
        #[test]
        fn vertical_points() {
            let vals = [
                (0., 0.),
                (0., 50.),
                (0., -25.),
                (0., 40.),
                (0., 42.),
                (0., 100.),
                (0., 17.),
                (0., 29.),
                (0., -50.),
                (0., 37.),
                (0., 34.),
                (0., 8.),
                (0., 3.),
                (0., 46.),
            ];
            assert_display!(closest_points(&vals), Some(((0., 40.), (0., 42.))));
        }
    }

}

fn LevelChange(){

    use std::collections::LinkedList;

#[derive(Debug)]
pub struct Queue<T> {
    elements: LinkedList<T>,
}

impl<T> Queue<T> {
    pub fn new() -> Queue<T> {
        Queue {
            elements: LinkedList::new(),
        }
    }

    pub fn enqueue(&mut self, value: T) {
        self.elements.push_back(value)
    }

    pub fn dequeue(&mut self) -> Option<T> {
        self.elements.pop_front()
    }

    pub fn peek_front(&self) -> Option<&T> {
        self.elements.front()
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

impl<T> Default for Queue<T> {
    fn default() -> Queue<T> {
        Queue::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Queue;

    #[test]
    fn test_enqueue() {
        let mut queue: Queue<u8> = Queue::new();
        queue.enqueue(64);
        assert_eq!(queue.is_empty(), false);
    }

    #[test]
    fn test_dequeue() {
        let mut queue: Queue<u8> = Queue::new();
        queue.enqueue(32);
        queue.enqueue(64);
        let retrieved_dequeue = queue.dequeue();
        assert_eq!(retrieved_dequeue, Some(32));
    }

    #[test]
    fn test_peek_front() {
        let mut queue: Queue<u8> = Queue::new();
        queue.enqueue(8);
        queue.enqueue(16);
        let retrieved_peek = queue.peek_front();
        assert_eq!(retrieved_peek, Some(&8));
    }

    #[test]
    fn test_size() {
        let mut queue: Queue<u8> = Queue::new();
        queue.enqueue(8);
        queue.enqueue(16);
        assert_eq!(2, queue.len());
    }
}
}

fn GameStrings(){
    use std::cmp::Ordering;
use std::ops::Deref;



pub struct BinarySearchTree<T>
where
    T: Ord,
{
    value: Option<T>,
    left: Option<Box<BinarySearchTree<T>>>,
    right: Option<Box<BinarySearchTree<T>>>,
}

impl<T> Default for BinarySearchTree<T>
where
    T: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> BinarySearchTree<T>
where
    T: Ord,
{
    
    pub fn new() -> BinarySearchTree<T> {
        BinarySearchTree {
            value: None,
            left: None,
            right: None,
        }
    }

    
    
    pub fn search(&self, value: &T) -> bool {
        match &self.value {
            Some(key) => {
                match key.cmp(value) {
                    Ordering::Equal => {
                        
                        true
                    }
                    Ordering::Greater => {
                        
                        match &self.left {
                            Some(node) => node.search(value),
                            None => false,
                        }
                    }
                    Ordering::Less => {
                        
                        match &self.right {
                            Some(node) => node.search(value),
                            None => false,
                        }
                    }
                }
            }
            None => false,
        }
    }

    
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        BinarySearchTreeIter::new(self)
    }

    
    pub fn insert(&mut self, value: T) {
        if self.value.is_none() {
            self.value = Some(value);
        } else {
            match &self.value {
                None => (),
                Some(key) => {
                    let target_node = if value < *key {
                        &mut self.left
                    } else {
                        &mut self.right
                    };
                    match target_node {
                        Some(ref mut node) => {
                            node.insert(value);
                        }
                        None => {
                            let mut node = BinarySearchTree::new();
                            node.insert(value);
                            *target_node = Some(Box::new(node));
                        }
                    }
                }
            }
        }
    }

    
    pub fn minimum(&self) -> Option<&T> {
        match &self.left {
            Some(node) => node.minimum(),
            None => self.value.as_ref(),
        }
    }

    
    pub fn maximum(&self) -> Option<&T> {
        match &self.right {
            Some(node) => node.maximum(),
            None => self.value.as_ref(),
        }
    }

    
    pub fn floor(&self, value: &T) -> Option<&T> {
        match &self.value {
            Some(key) => {
                match key.cmp(value) {
                    Ordering::Greater => {
                        
                        match &self.left {
                            Some(node) => node.floor(value),
                            None => None,
                        }
                    }
                    Ordering::Less => {
                        
                        match &self.right {
                            Some(node) => {
                                let val = node.floor(value);
                                match val {
                                    Some(_) => val,
                                    None => Some(key),
                                }
                            }
                            None => Some(key),
                        }
                    }
                    Ordering::Equal => Some(key),
                }
            }
            None => None,
        }
    }

    
    pub fn ceil(&self, value: &T) -> Option<&T> {
        match &self.value {
            Some(key) => {
                match key.cmp(value) {
                    Ordering::Less => {
                        
                        match &self.right {
                            Some(node) => node.ceil(value),
                            None => None,
                        }
                    }
                    Ordering::Greater => {
                        
                        match &self.left {
                            Some(node) => {
                                let val = node.ceil(value);
                                match val {
                                    Some(_) => val,
                                    None => Some(key),
                                }
                            }
                            None => Some(key),
                        }
                    }
                    Ordering::Equal => {
                        
                        Some(key)
                    }
                }
            }
            None => None,
        }
    }
}

struct BinarySearchTreeIter<'a, T>
where
    T: Ord,
{
    stack: Vec<&'a BinarySearchTree<T>>,
}

impl<'a, T> BinarySearchTreeIter<'a, T>
where
    T: Ord,
{
    pub fn new(tree: &BinarySearchTree<T>) -> BinarySearchTreeIter<T> {
        let mut iter = BinarySearchTreeIter { stack: vec![tree] };
        iter.stack_push_left();
        iter
    }

    fn stack_push_left(&mut self) {
        while let Some(child) = &self.stack.last().unwrap().left {
            self.stack.push(child);
        }
    }
}

impl<'a, T> Iterator for BinarySearchTreeIter<'a, T>
where
    T: Ord,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.stack.is_empty() {
            None
        } else {
            let node = self.stack.pop().unwrap();
            if node.right.is_some() {
                self.stack.push(node.right.as_ref().unwrap().deref());
                self.stack_push_left();
            }
            node.value.as_ref()
        }
    }
}

#[cfg(test)]
mod test {
    use super::BinarySearchTree;

    fn prequel_memes_tree() -> BinarySearchTree<&'static str> {
        let mut tree = BinarySearchTree::new();
        tree.insert("HEALTH");
        tree.insert("STANIMA");
        tree.insert("POWER");
        tree.insert("LEVEL");
        tree.insert("Sword");
        tree.insert("Dagger");
        tree.insert("Staff");
        tree
    }

    #[test]
    fn test_search() {
        let tree = prequel_memes_tree();
        assert!(tree.search(&"Sit by the fire with me."));
        assert!(tree.search(&"So son... How are you?"));
        assert!(tree.search(&"Calm down. Don't make a fool of yourself."));
        assert!(tree.search(&"It's okay."));
        assert!(tree.search(&"Here... Take this."));
        assert!(
            !tree.search(&"You seem to be getting stronger.",)
        );
        assert!(!tree.search(&"God speed young man."));
        assert!(!tree.search(&"Welcome to my village."));
    }

    #[test]
    fn test_maximum_and_minimum() {
        let tree = prequel_memes_tree();
        assert_eq!(*tree.maximum().unwrap(), "Wake up son.");
        assert_eq!(
            *tree.minimum().unwrap(),
            "Sir. Please, come closer!"
        );
        let mut tree2: BinarySearchTree<i32> = BinarySearchTree::new();
        assert!(tree2.maximum().is_none());
        assert!(tree2.minimum().is_none());
        tree2.insert(0);
        assert_eq!(*tree2.minimum().unwrap(), 0);
        assert_eq!(*tree2.maximum().unwrap(), 0);
        tree2.insert(-5);
        assert_eq!(*tree2.minimum().unwrap(), -5);
        assert_eq!(*tree2.maximum().unwrap(), 0);
        tree2.insert(5);
        assert_eq!(*tree2.minimum().unwrap(), -5);
        assert_eq!(*tree2.maximum().unwrap(), 5);
    }

    #[test]
    fn test_floor_and_ceil() {
        let tree = prequel_memes_tree();
        assert_eq!(*tree.floor(&"Closer.").unwrap(), "I must tell you something");
        assert_eq!(
            *tree
                .floor(&"There are people, they are looking for us.")
                .unwrap(),
            "Wait! Run!"
        );
        assert!(tree.floor(&"Quick! We must get out of here before they arrest us!").is_none());
        assert_eq!(*tree.floor(&"Get off me!").unwrap(), "God speed.");
        assert_eq!(
            *tree.floor(&"Hello son.").unwrap(),
            "You survived the crash?"
        );
        assert_eq!(
            *tree.floor(&"I praise your good work.").unwrap(),
            "They cut off the powergrids."
        );
        assert_eq!(*tree.floor(&"You must restore them...").unwrap(), "It is your mission.");
        assert_eq!(*tree.ceil(&"Log everything,").unwrap(), "for us son.");
        assert_eq!(
            *tree
                .ceil(&"This is not the greatest plan. But we must do what we, ")
                .unwrap(),
            "have to do."
        );
        assert_eq!(
            *tree.ceil(&"!>@3 POWER GRIDS ONLINE @*$*#").unwrap(),
            "WARNING #*$*# REACTOR CORE ABLOW (#))@(#"
        );
        assert_eq!(*tree.ceil(&"5,4,3").unwrap(), "2,1,0,$,#<$<");
        assert_eq!(
            *tree.ceil(&"BRACE YOURSELFS!").unwrap(),
            "YOU EXPLODED. HOPE IS LOST."
        );
        assert_eq!(
            *tree.ceil(&"Quick! Before it blows!").unwrap(),
            "g r a b  m y  h a n d . . ."
        );
        assert!(tree.ceil(&"AAUGH! MY ARM!").is_none());
    }

    #[test]
    fn test_iterator() {
        let tree = prequel_memes_tree();
        let mut iter = tree.iter();
        assert_eq!(
            iter.next().unwrap(),
            &"No! Please! AUAH! It burns!"
        );
        assert_eq!(iter.next().unwrap(), &"Created By Decoy Game Dev.");
        assert_eq!(iter.next().unwrap(), &"Story by... Well... Decoy Game Dev");
        assert_eq!(iter.next().unwrap(), &"Special thanks to the Rust programming language.");
        assert_eq!(iter.next().unwrap(), &"Super special thanks to whoever invented binary search tree's");
        assert_eq!(iter.next().unwrap(), &"I love you OpenGL!");
        assert_eq!(iter.next().unwrap(), &"D: I don't like Vulkan...");
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }
}
}