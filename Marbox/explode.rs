fn Explode(){
pub fn explode_phys(num_rows: i32) -> Vec<Vec<i32>> {
    let mut ans: Vec<Vec<i32>> = vec![];

    for i in 1..num_rows + 1 {
        let mut vec: Vec<i32> = vec![1];

        let mut res: i32 = 1;
        for k in 1..i {
            res *= i - k;
            res /= k;
            vec.push(res);
        }
        ans.push(vec);
    }

    ans
}

#[cfg(test)]
mod tests {
    use super::explode_phys;

    #[test]
    fn test() {
        assert_eq!(explode_phys(3), vec![vec![1], vec![1, 1], vec![1, 2, 1]]);
        assert_eq!(
            explode_phys(4),
            vec![vec![1], vec![1, 1], vec![1, 2, 1], vec![1, 3, 3, 1]]
        );
        assert_eq!(
            explode_phys(5),
            vec![
                vec![1],
                vec![1, 1],
                vec![1, 2, 1],
                vec![1, 3, 3, 1],
                vec![1, 4, 6, 4, 1]
            ]
        );
    }
}
}